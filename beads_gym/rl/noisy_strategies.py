import torch
import numpy as np


class GreedyStrategy:
    def __init__(self, bounds):
        self.low, self.high = bounds
        self.ratio_noise_injected = 0

    def select_action(self, model, state):
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        action = np.clip(greedy_action, self.low, self.high)
        return np.reshape(action, self.high.shape)


class NormalNoiseStrategy:
    def __init__(self, bounds, exploration_noise_ratio=0.1, exploration_noise_amplitude=None, ou_process=False):
        self.low, self.high = bounds
        
        if exploration_noise_ratio is None: assert exploration_noise_amplitude is not None
        if exploration_noise_amplitude is None: assert exploration_noise_ratio is not None
        self.exploration_noise_ratio = exploration_noise_ratio
        self.exploration_noise_amplitude = exploration_noise_amplitude
        self.ou_process = ou_process
        self.prev_noise = np.zeros(len(self.high))
        self.ratio_noise_injected = 0
        
    def select_action(self, model, state, max_exploration=False):
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
            
        noise = self.prev_noise if self.ou_process else np.zeros(len(self.high))
        if max_exploration:
            noise += np.random.normal(loc=0, scale=self.high, size=len(self.high))
        else:
            if self.exploration_noise_ratio is not None:
                noise += np.random.normal(loc=0, scale=1, size=len(self.high)) * np.abs(greedy_action) * self.exploration_noise_ratio
            elif self.exploration_noise_amplitude is not None:
                noise += np.random.normal(loc=0, scale=self.exploration_noise_amplitude, size=len(self.high))
            else:
                raise ValueError("No exploration noise specified")

        noisy_action = greedy_action + noise
        self.prev_noise = noise
        action = np.clip(noisy_action, self.low, self.high)
        
        self.ratio_noise_injected = np.mean(abs((greedy_action - action)/(self.high - self.low)))
        return action
    
    
class NormalNoiseDecayStrategy:
    def __init__(
        self,
        bounds,
        init_noise_ratio_mult=0.5, min_noise_ratio_mult=0.1,
        init_noise_ratio_add=0.5, min_noise_ratio_add=0.1,
        decay_steps=10000,
    ):
        self.t = 0
        self.low, self.high = bounds
        self.noise_ratio_mult = init_noise_ratio_mult
        self.init_noise_ratio_mult = init_noise_ratio_mult
        self.min_noise_ratio_mult = min_noise_ratio_mult
        self.noise_ratio_add = init_noise_ratio_add
        self.init_noise_ratio_add = init_noise_ratio_add
        self.min_noise_ratio_add = min_noise_ratio_add
        self.decay_steps = decay_steps
        self.ratio_noise_injected = 0

    def _noise_ratio_update(self):
        noise_ratio = 1 - self.t / self.decay_steps
        noise_ratio_mult = (self.init_noise_ratio_mult - self.min_noise_ratio_mult) * noise_ratio + self.min_noise_ratio_mult
        self.noise_ratio_mult = np.clip(noise_ratio_mult, self.min_noise_ratio_mult, self.init_noise_ratio_mult)
        noise_ratio_add = (self.init_noise_ratio_add - self.min_noise_ratio_add) * noise_ratio + self.min_noise_ratio_add
        self.noise_ratio_add = np.clip(noise_ratio_add, self.min_noise_ratio_add, self.init_noise_ratio_add)
        self.t += 1

    def select_action(self, model, state, max_exploration=False):
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
            
        noise = np.zeros(len(self.high))
        if max_exploration:
            noise += np.random.normal(loc=0, scale=self.high, size=len(self.high))
        else:
            mult_noise_scale = np.abs(greedy_action) * self.noise_ratio_mult
            noise += np.random.normal(loc=0, scale=mult_noise_scale, size=len(self.high))
            noise += np.random.normal(loc=0, scale=self.noise_ratio_add, size=len(self.high))

        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)
        
        self._noise_ratio_update()
        
        self.ratio_noise_injected = np.mean(abs((greedy_action - action)/(self.high - self.low)))
        return action
