import io
import numpy as np
from gym.spaces.box import Box

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from beads_gym.environment.environment_cpp import EnvironmentCpp
from beads_gym.beads.beads import Bead
from beads_gym.bonds.bonds import DistanceBond
from beads_gym.environment.reward.rewards import StayCloseReward


REWARD_BOTTOM = -1



def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class BeadsCartPoleEnvironment:
    def __init__(self):
        self.env_backend = EnvironmentCpp(0.01)
        bead_0 = Bead(0, [0, 0, 0], 1.0, True)
        bead_1 = Bead(1, [0, 0, 1], 1.0, True)
        self.env_backend.add_bead(bead_0)
        self.env_backend.add_bead(bead_1)
        
        distance_bond = DistanceBond(0, 1)
        self.env_backend.add_bond(distance_bond)
        
        reward_calculator = StayCloseReward({
            0: np.array([0, 0, 0]),
            1: np.array([0, 0, 1]),
        })
        self.env_backend.add_reward_calculator(reward_calculator)
        self.count = 0
        
        self.videos = []
        
    def reset(self):
        self.env_backend.reset()
        self.count = 0
        self.videos.append([])
        return self._state()
    
    def step(self, action):
        action = {0: action}
        partial_rewards = self.env_backend.step(action)
        reward = 1 + sum(partial_rewards)
        new_state = self._state()
        self.count += 1
        truncated = (self.count == 1000)
        if truncated:
            info = {"TimeLimit.truncated": truncated}
        else:
            info = {}
        return new_state, reward, (truncated or reward <= REWARD_BOTTOM), info
    
    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            plt.clf()
            grid_size_x, grid_size_y = 4, 3
            gs = gridspec.GridSpec(grid_size_x, grid_size_y)
            fig_x = 16
            fig_y = 9
            fig = plt.figure(
                figsize=(fig_x, fig_y),
                dpi=60,
                facecolor=(0.8, 0.8, 0.8),
            )
            
            ax0 = fig.add_subplot(gs[:grid_size_x, :grid_size_y], projection="3d", facecolor=(0.9, 0.9, 0.9))
            positions = np.r_[[bead.get_position() for bead in self.env_backend.get_beads()]]
            x, y, z = positions.T
            ax0.plot(x, y, z, "b", linewidth=3, label="bonds")
            ax0.scatter(positions[:, 0], positions[:, 1], positions[:, 2], linewidth=10, label="beads")
            ax0.set_xlim(-0.5, 0.5)
            ax0.set_ylim(-0.5, 0.5)
            ax0.set_zlim(0, 1.5)
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            
            img = Image.open(buf).convert("RGB")
            rgb_array = np.array(img)
            plt.close(fig)
            
            self.videos[-1].append(rgb_array)
            
            return rgb_array

    def _state(self):
        beads = self.env_backend.get_beads()
        vectorized = np.r_[
            [[bead.get_position(), bead.get_velocity(), bead.get_acceleration()] for bead in beads]
        ].flatten()
        state = np.r_[
            vectorized,
            np.linalg.norm(vectorized[:3] - vectorized[9:12]),
        ]
        return state
        
    def close(self):
        pass
    
    @property
    def spec(self):
        return None
    
    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}
    
    @property
    def reward_range(self):
        return Box(low=REWARD_BOTTOM, high=1.0, shape=(1,), dtype=np.float32)
        
    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(len(self._state()),), dtype=np.float32)
    
    @property
    def action_space(self):
        low = np.array([-5, -5, -5], dtype=np.float32)
        high = np.array([5, 5, 35], dtype=np.float32)
        return Box(low=low, high=high, shape=(3,), dtype=np.float32)

    def seed(self, seed=None):
        seed_everything(seed)
