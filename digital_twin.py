import os
import argparse
import gym
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from envs.digital_twin_env import DigitalTwinEnv


POLICY_NUMBER = 4
SEED = 7
ALGORITHM_NAME = 'PPO'
EVAL_FREQ = 10000

class DigitalTwin():
    '''Driver Class for Offline RL Training a DigitalTwin Ship'''
    
    def __init__(self, 
                 mode:str='train', 
                 best_model:bool=False, 
                 verbose_env:bool=False) -> None:
        '''Constructor for DigitalTwin Class'''
        
        # Tensorboard logging path
        self._log_path = os.path.join('policies', 'all_policy_'+str(POLICY_NUMBER), 'Logs')
        # Model Path
        self._model_path = os.path.join('policies', 'all_policy_'+str(POLICY_NUMBER), ALGORITHM_NAME + '_policy_'+str(POLICY_NUMBER))
        # Load Best Model
        if best_model:
            self._model_path = os.path.join('policies', 'all_policy_'+str(POLICY_NUMBER), 'best_model')
        # Model Directory
        self._model_directory = os.path.join('policies', 'all_policy_'+str(POLICY_NUMBER))
        
        #Initialize Open AI Gym environment. A
        self._env = DigitalTwinEnv(mode=mode, 
                                   verbose_env=verbose_env)
        
        # Timesteps based on number of data observations
        self._total_timesteps = self._env.get_num_rows() - 10000 # Training usually doesn't stop exactly at timesteps specified

        if mode == 'train':
            self.train()
        
        if mode == 'test':
            self.test()

    def train(self):
        '''Train Reinforcement Learning Policy'''
        
        # Add Wrappers for efficiency and callback function
        # self._env = Monitor(self._env, self._model_directory)
        # self._env = DummyVecEnv([lambda: self._env])
        
        #Initialize Model
        model = self.algorithm_class('MlpPolicy',
                                      self._env,
                                      verbose=1,
                                      tensorboard_log=self._log_path,
                                      seed=SEED)
        
        # Create callback to evaluate performance of model and save the best model
        eval_callback = EvalCallback(self._env, 
                                     best_model_save_path=self._model_directory,
                                     log_path=self._model_directory, 
                                     eval_freq=EVAL_FREQ, 
                                     deterministic=True)
        
        # Start Training Model
        model.learn(total_timesteps=self._total_timesteps,
                    callback=eval_callback)
        
        # Save Trained Model
        model.save(self._model_path)
        
        print('>> Model Saved <<')
    
    def test(self):
        '''Test a Saved Reinforcement Learning Policy'''
        model = self.algorithm_class.load(self._model_path, env=self._env)
        
        obs = self._env.reset()

        for _ in range(100):
            
            print('Observation:')
            self._env.display(obs)

            index_to_modify = input('Index to modify: ')
            if index_to_modify.isnumeric():
                index_to_modify = int(index_to_modify)
                perturbation_amount = input('Perturbation: ')
                obs[index_to_modify] = perturbation_amount
            
            action, _ = model.predict(obs, deterministic=True)
            
            print('Action:')
            self._env.display(action)

            obs, _, _, _ = self._env.step(action)
            
            another_observation = input('Next? [Y/n] ')
            if not another_observation.lower() == 'y':
                break
    
    @property
    def algorithm_class(self):
        '''Return the Stable-Baseliens3 Algorithm Class'''
        
        if ALGORITHM_NAME == 'PPO':
            return PPO
        elif ALGORITHM_NAME == 'DDPG':
            return DDPG

        raise ValueError('Unsupported Agorithm')
    
if __name__ == '__main__':
    # Input commands
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', dest='mode', type=str, default='train')
    arg_parser.add_argument('--verbose_env', dest='verbose_env', action='store_true', default=False)
    arg_parser.add_argument('--best_model', dest='best_model', action='store_true', default=False)
    args = arg_parser.parse_args()
    
        
    run = DigitalTwin(mode=args.mode, 
                      best_model=args.best_model, 
                      verbose_env=args.verbose_env)
