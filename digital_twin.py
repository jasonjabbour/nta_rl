import os
import argparse
import gym
from stable_baselines3 import DDPG, PPO
from envs.digital_twin_env import DigitalTwinEnv

POLICY_NUMBER = 2
SEED = 7
ALGORITHM_NAME = 'DDPG'
DISCOUNT_FACTOR = .1

class DigitalTwin():
    '''Driver Class for Offline RL Training a DigitalTwin Ship'''
    
    def __init__(self, 
                 mode:str='train') -> None:
        '''Constructor for DigitalTwin Class'''
        
        # Tensorboard logging path
        self._log_path = os.path.join('policies', 'all_policy_'+str(POLICY_NUMBER), 'Logs')
        # Model Path
        self._model_path = os.path.join('policies', 'all_policy_'+str(POLICY_NUMBER), ALGORITHM_NAME + '_policy_'+str(POLICY_NUMBER))
        
        #Initialize Open AI Gym environment
        self._env = DigitalTwinEnv(mode=mode)
        
        # Timesteps based on number of data observations
        self._total_timesteps =  self._env.num_rows - 5000 # Training usually doesn't stop exactly at timesteps specified
        self._total_timesteps = 1000
        if mode == 'train':
            self.train()
        
        if mode == 'test':
            self.test()


    def train(self):
        '''Train Reinforcement Learning Policy'''
        
        model = self.algorithm_class('MlpPolicy',
                                      self._env,
                                      verbose=1,
                                      tensorboard_log=self._log_path,
                                      seed=SEED, 
                                      gamma=DISCOUNT_FACTOR)
        model.learn(total_timesteps=self._total_timesteps)
        model.save(self._model_path)
        print('>> Model Saved <<')
    
    def test(self):
        '''Test a Saved Reinforcement LEarning Policy'''
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
    args = arg_parser.parse_args()
    
        
    run = DigitalTwin(mode=args.mode)
