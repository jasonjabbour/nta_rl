import os 
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pandas as pd
import numpy as np
import time
import gym 
from gym import spaces

from envs.digital_twin_config import EnvironmentConfig
from envs.attackers.naive_attacker import NaiveAttacker

class DigitalTwinEnv(gym.Env):
    '''Open AI Gym Environment for Ship Digital Twin Offline Learning'''
    
    def __init__(self,
                 mode:str = 'train', 
                 verbose_env:bool = False) -> None:
        '''Constructor for DigitalTwinEnv Class'''
        
        # Load Configuration Parameters
        self.load_config_parameters()
                
        # Set number of history observations to store
        self._history_buffer_size = 5
        
        # Indicate where in the data you current are located
        self._simulation_step_pointer = 0 + self._history_buffer_size
        
        # Build Action Space
        self.build_action_space()
        
        # Build Observational Space
        self.build_observational_space()
        
        # Read in dataset
        self.read_data()
        self.data_summary()
        
        # Initialize the attacker module
        self._attacker = NaiveAttacker('NaiveAttacker', verbose=verbose_env)
        
        # Mode of the environment train or test
        self._mode = mode
        
        self.reset()
            
    def load_config_parameters(self):
        '''Load Config Parameters into this class'''
        env_config = EnvironmentConfig()
        self._data_dir = env_config.data_dir
        self._data_file_names = env_config.data_file_names
        self._test_data_file_names = env_config.test_data_file_names
        self._features = env_config.features
        self._include_feature_as_action = env_config.include_features_as_action
        self._include_feature_as_observation = env_config.include_features_as_observation
        self._steps_per_episode = env_config.steps_per_episode
        self._attack_detected_bonus = env_config.attack_detected_bonus
        self._attack_TP_threshold = env_config.attack_TP_threshold
        self._attack_FN_threshold = env_config.attack_FN_threshold
        self._attack_FP_threshold = env_config.attack_FP_threshold
        self._attack_TN_threshold = env_config.attack_TN_threshold
        
    def build_action_space(self):
        '''Build Action Space'''
        
        # Get number of features
        self._included_features_as_action_num = self.get_num_action_features()
        
        # Set lower and upper bound to [0,1] for all included features
        action_upper_bound = [1]*(self._included_features_as_action_num+1)
        action_lower_bound = [0]*(self._included_features_as_action_num+1)
        
        # Create Box Action Space
        self.action_space = spaces.Box(low=np.array(action_lower_bound), 
                                                high=np.array(action_upper_bound), 
                                                dtype=np.float32)
        
            
    def build_observational_space(self):
        '''Build Observational Space'''
        
        # Get number of features
        self._included_features_in_obs_num = self.get_num_observation_features()
        
        # m features * (n history + 1 current observation)
        observation_length = self._included_features_in_obs_num*(self._history_buffer_size+1)
        
        # Set lower and upper bound to [0,1] for all included features and history buffer
        obs_upper_bound = [np.inf]*observation_length
        obs_lower_bound = [-np.inf]*observation_length
        
        # Create Box Observational Space
        self.observation_space = spaces.Box(low=np.array(obs_lower_bound),
                                            high=np.array(obs_upper_bound), 
                                            dtype=np.float32)
    
    
    def step(self, action):
        '''Using the action predicted step the simulation'''
        
        # Increment step and pointer
        self._episode_step_num+=1
        self._simulation_step_pointer+=1
        
        # get the reward for the previous observation
        reward = self._reward(action)
        
        # get the new observattion
        observation = self._get_observation()
        
        # Determine if episode should end
        done = self._terminate_episode()
        
        info = {}

        return np.array(observation), reward, done, info
    
    def _reward(self, action):
        '''Calculate reward for the given action
        
        Args:
            action: the action space consists of continuous confidence values for each observation 
                along with an action to indicate whether an attack is in progress. 
        '''
        
        bonus = 10
        
        # Determine if we are currently under attack (T/F)
        under_attack = self._attacker.get_attack_in_progress()
        # Indices under attack (if any)
        indices_under_attack_in_previous_step = self._attacker.get_indices_under_attack()
        
        reward_lst = []     
        # All actions except last action since the last action is a discrete attack indicator   
        for index in range(len(action)-1):
            # Index was perturbed penalize for high confidence level
            if index in indices_under_attack_in_previous_step:
                reward_lst.append(-action[index])
            # Index was not previously perturbed maximize hight confidence level
            else:
                reward_lst.append(action[index])
        
        # If attack detected correctly (TP)
        if under_attack and (action[-1] > self._attack_TP_threshold):
            # add a bonus reward
            reward_lst.append(self._attack_detected_bonus)
            
            # since correctly detected attack, end the attack and plan a new one
            self._attacker.end_attack(plan_next_attack=True)
            
        # If under attack and not detected (FN) or not underattack and detected attack (FP)
        elif (under_attack and (action[-1] <= self._attack_FN_threshold)) or ((not under_attack) and (action[-1] >= self._attack_FP_threshold)):
            # penalize
            reward_lst.append(-self._attack_detected_bonus)
            
        # If not under attack and no attack detected (TN)
        elif (not under_attack) and (action[-1] <= self._attack_TN_threshold):
            # add bonus reward
            reward_lst.append(self._attack_detected_bonus)

        # Add up all rewards
        reward = sum(reward_lst) 

        return reward
    
    def _get_observation(self):
        '''Create observation to feed agent'''
        
        # Get new row of data
        row = self._df.iloc[self._simulation_step_pointer].values
                
        # Select observations
        obs = []
        for feature, included in zip(row, self._include_feature_as_observation):
            if included:
                obs.append(feature)

        # Send observation to attacker module
        self._processed_obs = obs
        if not self._mode == 'test': 
            self._processed_obs = self._attacker.on_step(obs)
            
        # Add the real previous observations history to this processed observation
        self._proccessed_obs_with_history = self.add_history(self._processed_obs)
                
        return np.array(self._proccessed_obs_with_history)

    def add_history(self, obs):
        '''Select previous rows from dataframe and add to current observation'''
        
        # Get names of columns to include in observation
        included_columns_as_features = self.get_observation_feature_names()
        
        # Make a Dataframe with only these columns
        included_cols_df = self._df[included_columns_as_features]
        
        # Get all n history excluding current row. Current row is the obs passed in that could have been perturbed
        real_history_observations = np.array(included_cols_df.iloc[self._simulation_step_pointer-self._history_buffer_size:self._simulation_step_pointer].values)
        # Flatten rows into 1D list
        real_history_observations = real_history_observations.flatten()
        
        # Return a list of the observation along with history
        return np.concatenate((obs, real_history_observations))
        
    
    def _terminate_episode(self):
        '''Determine whether to end epside'''
        if self._episode_step_num >= self._steps_per_episode:
            return True
        return False

    def reset(self):
        '''Reset in preparation for new episode'''
        
        # Step number in episode
        self._episode_step_num = 0  
        
        # Reset attacker 
        self._attacker.on_reset()
        
        return self._get_observation()

    def read_data(self, mode='train'):
        '''Read in all data as one pandas dataframe'''
        
        data_files = self._data_file_names
        if mode == 'test':
            data_files = self._test_data_file_names
            
        dataframe_lst = []
        first_csv = True
        # Read in each csv file
        for file_name in data_files:
            # Only load header for the first dataframe
            if first_csv:
                df = pd.read_csv(self._data_dir + file_name+'.csv', index_col=None)
                first_csv = False
            else:
                df = pd.read_csv(self._data_dir + file_name+'.csv', index_col=None, header=0)
            dataframe_lst.append(df)
        
        # Concatinate all dataframes into one big dataframe
        self._df = pd.concat(dataframe_lst, axis=0, ignore_index=True)

    def get_num_action_features(self):
        '''Return the number of features that will have a corresponding action'''
        
        included_features_as_action_num = 0
        for i in self._include_feature_as_action:
            if i == 1:
                included_features_as_action_num+=1
        
        return included_features_as_action_num
    
    def get_num_observation_features(self):
        '''Return the number of features that will be passed in as an observation'''
        
        included_observation_num = 0
        for i in self._include_feature_as_observation:
            if i == 1:
                included_observation_num+=1
        
        return included_observation_num
    
    def get_observation_feature_names(self):
        '''Return the column names of the features that will be passed in as an observation'''
        included_feature_names = []
        for i, included in enumerate(self._include_feature_as_observation):
            if included == 1:
                included_feature_names.append(self._features[i])
        
        return included_feature_names
                
    def data_summary(self):
        
        # subtract 1 to account for header 
        self._num_rows = self._df.shape[0] - 1
        self._num_cols = self._df.shape[1] - 1
        
        #print(self._df.head())
                
        # print(self._df.iloc[98078])
        # print(self._df['position_x'].max())
    
    def get_action_space_sample(self):
        return self.action_space.sample()
    
    def get_observational_space_sample(self):
        return self.observation_space.sample()        
    
    def render(self):
        pass

    def close(self):
        pass
    
    def display(self, features):
        #Assume observation in feature is in order!
        features_num = self.get_num_observation_features()
        count = 0
        for feature in zip(features):
            if count >= features_num:
                break
            print('['+str(count)+'] \t' + self._features[count] + ': ' + str(round(feature[0],2)))
            count+=1
                    

    def get_num_rows(self):
        return self._num_rows
    
    def get_num_cols(self):
        return self._num_cols
    





