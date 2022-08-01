import os 
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


import random
import numpy as np 
from envs import digital_twin_config
from envs.attackers import attacker

# At the latest start the attack at the last 20% of the episode
LATEST_ATTACK_START_TIME = .2

class NaiveAttacker(attacker.Attacker):
    '''A naive attacker that perturbs observations until it is detected.'''
    
    def __init__(self, 
                 name: str, 
                 cooldown: int = 20, 
                 verbose: bool = False):
        '''NaiveAttacker Constructor
        
        
        Args:
            name: name of the attack
            cooldown: number of timesteps to wait before launching another attack
            verbose: output logs about what the attack is doing to console 
        '''
        

        # Read in configuration parameters
        self._env_config = digital_twin_config.EnvironmentConfig()

        # Total timesteps in an episode
        self._steps_per_episode = self._env_config.steps_per_episode
        
        # Cooldown timesteps between attacks
        self._cooldown = cooldown
        
        # Set if attacker will be verbose
        self._verbose = verbose
        
        # State of attack
        self._attack_in_progress = False 
        
        # Keep track of current step within episode
        self._episode_timestep_counter = 0 
        
        # Timestep to attack
        self._planned_time_2_attack = -1 
        
        # Pass arguments to super class
        super(NaiveAttacker, self).__init__(
            name=name
        )
        
        # Start the attack driver
        self.attack_driver()
        
    def attack_driver(self):
        '''Plans the timestep when the attack should start'''
        
        # Plan the timestep to launch an attack
        if not self._attack_in_progress:
            # Must select a timestep at least x% before episode ends
            self._planned_time_2_attack = random.randint(0, self._steps_per_episode - int(LATEST_ATTACK_START_TIME*self._steps_per_episode)) 
        
    def get_attack_in_progress(self):
        '''Return whether or not an attack is in progress'''
        return self._attack_in_progress
        
    def permission_to_start_attack(self) -> bool:
        '''Compare the attack start timestep time with the current timestep. 
            If attack is not planned self._planned_time_2_attack will have a default
            value of -1. 
        '''
        if (self._planned_time_2_attack >= self._episode_timestep_counter):
            return True
        return False
             
    def on_step(self, obs):
        ''' A callback function for the step event.
        
        Args:
            obs: observation at the current timetep
        '''
        
        # Increase the counter
        self._episode_timestep_counter+=1
        
        # If the attack has not started check and it's time to launch attack
        if (not self._attack_in_progress) and self.permission_to_start_attack():
            # Start the attack
            self.start_attack(obs)
            
        # If attack is in progress modify the observation
        if self._attack_in_progress:
            return self.process_observation(obs)
        
        # If attack is not in progress do not modify observation
        return obs
    
    
    def start_attack(self, obs):
        '''Start attack by randomly choosing which observation indices should be attacked'''
        
        self._num_observations = len(obs)
        
        # Select how many observations to perturb by generating multiple options
        num_observations_to_perturb_options = [np.random.uniform(0, self._num_observations) for i in range(self._num_observations)]
        
        # Select the option that is the minumum number of observations to perturb
        num_observations_to_perturb = int(min(num_observations_to_perturb_options))
        
        # Randomly choose which indices to perturb 
        indices_under_attack_optional = [np.random.randint(0, self._num_observations-1) for i in range(num_observations_to_perturb)]   
        
        self._indices_under_attack = []
        # Check if this observation is allowed to be perturbed and not already perturbed
        for index in indices_under_attack_optional:
            if self._env_config.include_features_as_action[index] and (index not in self._indices_under_attack):
                self._indices_under_attack.append(index)
                
        # Initiate this attack
        self._attack_in_progress = True
        
        if self._verbose:
            print('Attack initiated!')

    def process_observation(self, obs):
        '''Randomly choose a perturbation amount to attack each of the chosen indices
        
        Args:
            obs: An OpenAI Gym observation
        Returns:
            attacked_obs: an attacked observation
        '''
        
        # Create copy of observation
        attacked_obs = obs.copy()

        perturbation_amount = 0 
        indices_under_attack = self.get_indices_under_attack()
        
        # Go through chosen indices of observational space to perturb
        for index in indices_under_attack:
                # Randomly select min and max
                max_perturbation_amount = np.random.randint(1, 1000)
                min_perturbation_amount = -np.random.randint(1, 1000)
                # Select pertubraiton amount
                perturbation_amount = np.random.randint(min_perturbation_amount, max_perturbation_amount)
                # Make sure to choose a number that is not zero
                while perturbation_amount == 0: 
                    perturbation_amount = np.random.randint(min_perturbation_amount, max_perturbation_amount)
                # Randomly select perturbation amount using random max and min
                attacked_obs[index] = attacked_obs[index] + perturbation_amount
    
        return attacked_obs

    def get_indices_under_attack(self):
        '''Return the indices currently chosen to be atacked'''
        if self._indices_under_attack:
            return self._indices_under_attack
        return list()
        
    def on_reset(self):
        '''Callback function for the reset event''' 
        # Reset attack and episode 
        self._attack_in_progress = False
        self._episode_timestep_counter = 0  
        self._planned_time_2_attack = -1 
        
        # Plan next attack
        self.attack_driver()
    

    def end_attack(self, plan_next_attack=False):
        ''' End attack '''
        # Reset attack 
        self._attack_in_progress = False
        self._planned_time_2_attack = -1 
        
        # Plan Next attack
        if plan_next_attack:
            self.attack_driver()
        
        if self._verbose:
            print('Attack ended.')
                  
    
if __name__ == '__main__':
    attack = NaiveAttacker('NaiveAttacker', verbose=True)