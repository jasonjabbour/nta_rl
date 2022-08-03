import os 
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


import random
import numpy as np 
from envs import digital_twin_config
from envs.attackers import attacker


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
        self._attack_scheduled = False
        
        # Keep track of current step within episode
        self._episode_timestep_counter = 0 
        
        # Timestep to attack
        self._planned_time_2_attack = -1 
        
        # Start off with no indices under attack
        self._indices_under_attack = []
        
        # Pass arguments to super class
        super(NaiveAttacker, self).__init__(
            name=name
        )
        
        # Start the attack driver
        self.attack_driver()
        
    def attack_driver(self):
        '''Plans the timestep when the attack should start'''
        
        # Plan the timestep to launch an attack
        if not self._attack_scheduled:
            self._planned_time_2_attack = random.randint(self._episode_timestep_counter, self._steps_per_episode)
            # Now the attack time has been scheduled
            self._attack_scheduled = True
            
    def get_attack_in_progress(self):
        '''Return whether or not an attack is in progress'''
        return self._attack_in_progress

    def get_attack_scheduled(self):
        '''Return whether or not an attack has been scheduled'''
        return self._attack_scheduled

    def get_planned_time_2_attack(self):
        '''Return the timestep attack will be launched'''
        return self._planned_time_2_attack
        
    def permission_to_start_attack(self) -> bool:
        '''Compare the attack start timestep time with the current timestep. 
            If attack is not planned self._planned_time_2_attack will have a default
            value of -1. Gaining permission to start also requires that the attack has
            been scheduled and the attack is not already in progress. 
        '''
        # Attack must be scheduled and not in progress
        if self._attack_scheduled and (not self._attack_in_progress):
            # Scheduled time must be past current time
            if (self._episode_timestep_counter >= self._planned_time_2_attack):
                return True
        return False
             
    def on_step(self, obs):
        ''' A callback function for the step event.
        
        Args:
            obs: observation at the current timetep
        '''
                
        # Check whether it's time to laucn
        if self.permission_to_start_attack():
            # Start the attack
            self.start_attack(obs)
            
        # If attack is in progress modify the observation
        if self._attack_in_progress:
            obs = self.process_observation(obs)
        
        # Increase the counter
        self._episode_timestep_counter+=1
        
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
        return self._indices_under_attack
        
    def on_reset(self):
        '''Callback function for the reset event''' 
        # Reset attack  
        self.reset_attack()
    
        # Reset episode
        self.reset_episode()

        # Plan next attack
        self.attack_driver()
    

    def end_attack(self, plan_next_attack=False):
        ''' End attack 
        
        Args:
            plan_next_attack: Boolean value indicating whether
                or not to schedule another attack
        '''
        # Reset attack 
        self.reset_attack()
        
        # Plan Next attack
        if plan_next_attack:
            self.attack_driver()
        
        if self._verbose:
            print('Attack ended.')
                  
    def reset_attack(self):
        '''Reset all parameters correlated with an attack'''
        self._attack_scheduled = False
        self._attack_in_progress = False
        self._planned_time_2_attack = -1 
        self._indices_under_attack = []
    
    def reset_episode(self):
        '''Reset all parameters correlated with an episode'''
        self._episode_timestep_counter = 0  
        
if __name__ == '__main__':
    attack = NaiveAttacker('NaiveAttacker', verbose=True)