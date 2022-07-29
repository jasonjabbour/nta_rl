import pandas as pd
import numpy as np
import random
import gym 
from gym import spaces

DATA_DIR = 'data_collection/data/'
DATA_FILE_NAMES = ['digitaltwin_data_300waypoints_1ep_1',
                   'digitaltwin_data_300waypoints_1ep_1_wind4', 
                   'digitaltwin_data_620waypoints_1ep_wind5_maxcoord150']
# TEST_DATA_FILE_NAMES = ['digitaltwin_data_5waypoints_1ep_wind4_testdata']
TEST_DATA_FILE_NAMES = ['digitaltwin_data_300waypoints_1ep_1']

FEATURES = ['position_x','position_y','position_z',
            'orientation_x', 'orientation_y', 'orientation_z',
            'position_rate_x', 'position_rate_y', 'position_rate_z', 
            'orientation_rate_x', 'orientation_rate_y', 'orientation_rate_z',
            'wind_speed', 'engine_velocity_command', 'rudder_angle',
            'waypoint_x', 'waypoint_y', 'reached_waypoint','waypoint_counter']

INCLUDE_FEATURE_AS_ACTION = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0)
INCLUDE_FEATURE_AS_OBSERVATION = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0) #17 features in observation space

STEPS_PER_EPISODE = 100

class DigitalTwinEnv(gym.Env):
    '''Open AI Gym Envinrmnet for Ship Digital Twin Offline Learning'''
    
    def __init__(self,
                 mode:str='train') -> None:
        '''Constructor for DigitalTwinEnv Class'''
                
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
                
        self._mode = mode
        self._indices_perturbed_in_last_step = []
        self.reset()
                
    def build_action_space(self):
        '''Build Action Space'''
        
        # Get number of features
        self._included_features_as_action_num = self.get_num_action_features()
        
        # Set lower and upper bound to [0,1] for all included features
        action_upper_bound = [1]*self._included_features_as_action_num
        action_lower_bound = [0]*self._included_features_as_action_num
        
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
        
        # get the reward for the previous observation
        reward = self._reward(action)
        
        # get the new observattion
        observation = self._get_observation()
        
        # Increment step and pointer
        self._episode_step_num+=1
        self._simulation_step_pointer+=1
        
        # Determine if episode should end
        done = self._terminate_episode()
        
        info = {}

        return np.array(observation), reward, done, info
    
    def _reward(self, action):
        '''Calculate reward for the given action'''
        
        reward_lst = []        
        for index in range(len(action)):
            # Index was perturbed penalize for high confidence level
            if index in self._indices_perturbed_in_last_step:
                reward_lst.append(-action[index])
            # Index was not previously perturbed maximize hight confidence level
            else:
                reward_lst.append(action[index])

        # Add up all rewards
        reward = sum(reward_lst)   

        return reward
    
    def _get_observation(self):
        '''Create observation to feed agent'''
        
        # Get new row of data
        row = self._df.iloc[self._simulation_step_pointer].values
                
        # Select observations
        obs = []
        for feature, included in zip(row, INCLUDE_FEATURE_AS_OBSERVATION):
            if included:
                obs.append(feature)

        self._processed_obs = obs
        if not self._mode == 'test': 
            self._processed_obs = self.process_observation(obs) 
            
        # Add the real previous observations history to this processed observation
        self._proccessed_obs_with_history = self.add_history(self._processed_obs)
                
        return np.array(self._proccessed_obs_with_history )

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
        

    def process_observation(self, obs):
        '''Randomly choose which observation and how many observation to randomly perturb'''
        
        processed_obs = obs.copy()
        
        num_observations = len(processed_obs)
        
        # Select how many observations to perturb by generating multiple options
        num_observations_to_perturb_options = [np.random.uniform(0, num_observations) for i in range(num_observations)]
        
        # Select the option that is the minumum number of observations to perturb
        num_observations_to_perturb = int(min(num_observations_to_perturb_options))
        
        # Randomly choose which indices to perturb
        indices_to_perturb_optional = [np.random.randint(0, num_observations-1) for i in range(num_observations_to_perturb)]   
        
        perturbation_amount = 0 
        self._indices_perturbed_in_last_step = []
        
        # Go through chosen indices of observational space to perturb
        for index in indices_to_perturb_optional:
            # Check if this observation is allowed to be perturbed and not already perturbed
            if INCLUDE_FEATURE_AS_ACTION[index] and (index not in self._indices_perturbed_in_last_step):
                
                # Randomly select min and max
                max_perturbation_amount = np.random.randint(1, 1000)
                min_perturbation_amount = -np.random.randint(1, 1000)
                # Select pertubraiton amount
                perturbation_amount = np.random.randint(min_perturbation_amount, max_perturbation_amount)
                # Make sure to choose a number that is not zero
                while perturbation_amount == 0: 
                    perturbation_amount = np.random.randint(min_perturbation_amount, max_perturbation_amount)
                # Randomly select perturbation amount using random max and min
                processed_obs[index] = processed_obs[index] + perturbation_amount
                
                self._indices_perturbed_in_last_step.append(index)          
            
        return processed_obs
    
    def _terminate_episode(self):
        '''Determine whether to end epside'''
        if self._episode_step_num >= STEPS_PER_EPISODE:
            return True
        return False

    def reset(self):
        '''Reset in preparation for new episode'''
        
        # Step number in episode
        self._episode_step_num = 0  
        
        return self._get_observation()

    def read_data(self, mode='train'):
        '''Read in all data as one pandas dataframe'''
        
        data_files = DATA_FILE_NAMES
        if mode == 'test':
            data_files = TEST_DATA_FILE_NAMES
            
        dataframe_lst = []
        first_csv = True
        # Read in each csv file
        for file_name in data_files:
            # Only load header for the first dataframe
            if first_csv:
                df = pd.read_csv(DATA_DIR + file_name+'.csv', index_col=None)
                first_csv = False
            else:
                df = pd.read_csv(DATA_DIR + file_name+'.csv', index_col=None, header=0)
            dataframe_lst.append(df)
        
        # Concatinate all dataframes into one big dataframe
        self._df = pd.concat(dataframe_lst, axis=0, ignore_index=True)

    def get_num_action_features(self):
        '''Return the number of features that will have a corresponding action'''
        
        included_features_as_action_num = 0
        for i in INCLUDE_FEATURE_AS_ACTION:
            if i == 1:
                included_features_as_action_num+=1
        
        return included_features_as_action_num
    
    def get_num_observation_features(self):
        '''Return the number of features that will be passed in as an observation'''
        
        included_observation_num = 0
        for i in INCLUDE_FEATURE_AS_OBSERVATION:
            if i == 1:
                included_observation_num+=1
        
        return included_observation_num
    
    def get_observation_feature_names(self):
        '''Return the column names of the features that will be passed in as an observation'''
        included_feature_names = []
        for i, included in enumerate(INCLUDE_FEATURE_AS_OBSERVATION):
            if included == 1:
                included_feature_names.append(FEATURES[i])
        
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
            print('['+str(count)+'] \t' + FEATURES[count] + ': ' + str(round(feature[0],2)))
            count+=1
                    

    def get_num_rows(self):
        return self._num_rows
    
    def get_num_cols(self):
        return self._num_cols
    







