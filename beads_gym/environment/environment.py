import numpy as np
from gym.spaces.box import Box

from beads_gym.environment.environment_cpp import EnvironmentCpp


class Environment:
    def __init__(self):
        self.env_backend = EnvironmentCpp()
        
    def reset(self):
        return self.env_backend.reset()
    
    def step(self, action):
        self.env_backend.step(action)
        reward = self.env_backend.get_reward()
        beads = self.env_backend.get_beads()
        states = np.r_[[[bead.get_position(), bead.get_velocity(), bead.get_acceleration()] for bead in self.env_backend.get_beads()]]
        return reward
        
    @property
    def observation_space(self):
        return Box(low=-100, high=100, shape=(3,), dtype=np.float32)
    
    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(3,), dtype=np.float32)
