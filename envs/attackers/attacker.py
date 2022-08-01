'''An attacker prototype class'''

import numpy as np

class Attacker(object):
    '''An attacker prototype class'''
    
    def __init__(self, 
                 name: str):
        '''A basic constructor for an attacker
        
        
        Args:
            name: name of the attack
        '''
        
        self._name = name
        self._attack_in_progress = False
    
    def get_name(self) -> str:
        '''Name of the attacker'''
        return self._name

    def get_attack_in_progress(self) -> bool:
        '''Whether or not an attack is in progress'''
        return self._attack_in_progress
    
    def set_attack_in_progress(self, status):
        '''Sets the status of the attack
        
        Args:
            status: bool value whether or not an attack is in progress
        '''
        self._attack_in_progress = status
        
    def end_attack(self):
        ''' End attack by no longer changing observations.'''
        pass
    
    def start_attack(self):
        ''' Start attack by changing observations'''
        pass
    
    def attack_detected(self): 
        ''' Takes necessary steps when an attack has been detected by an agent.  '''
        pass
    
    def attack_driver(self):
        ''' Driver to determine when to start and end an attack'''
        pass
    
    def on_reset(self, env):
        '''Callback function for the reset event
        
        Args:
            env: the environment that invokes this call back function
        '''        
        pass
    
    def on_step(self, env):
        ''' A callback function for the step event.
        
        Args:
            env: the environment that invokes this call back function
        '''
        pass
    