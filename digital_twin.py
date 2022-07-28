import os
import argparse
import gym
from stable_baselines3 import PPO
from envs.digital_twin_env import DigitalTwinEnv

POLICY_NUMBER = 1
SEED = 7

class DigitalTwin():
    '''Driver Class for Offline RL Training a DigitalTwin Ship'''
    
    def __init__(self, 
                 mode:str='train') -> None:
        '''Constructor for DigitalTwin Class'''
        
        # Tensorboard logging path
        self._log_path = os.path.join('policies', 'all_policy_'+str(POLICY_NUMBER), 'Logs')
        # Model Path
        self._model_path = os.path.join('policies', 'all_policy_'+str(POLICY_NUMBER), 'policy_'+str(POLICY_NUMBER))
        
        #Initialize Open AI Gym environment
        self._env = DigitalTwinEnv()
        
        # Timesteps based on number of data observations
        self._total_timesteps =  self._env.num_rows - 4961 # 160,000 timesteps
        
        
        if mode == 'train':
            self.train()
        
        if mode == 'test':
            self.test()


    def train(self):
        model = PPO('MlpPolicy', self._env, verbose=1, tensorboard_log=self._log_path, seed=SEED)
        model.learn(total_timesteps=self._total_timesteps)
        model.save(self._model_path)
        print('>> Model Saved <<')
    
    def test(self):
        model = PPO.load(model, env=self._env)
        
        action = model.predict(obs, deterministic=True)

if __name__ == '__main__':
    # Input commands
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', dest='mode', type=str, default='train')
    args = arg_parser.parse_args()
    
        
    run = DigitalTwin(mode=args.mode)
