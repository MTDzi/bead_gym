import io
import math
import numpy as np
from gym.spaces.box import Box
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from beads_gym.environment.environment_cpp import EnvironmentCpp
from beads_gym.beads.beads import Bead
from beads_gym.bonds.bonds import DistanceBond
from beads_gym.environment.reward.rewards import StayCloseReward


class BeadsQuadCopterEnvironment:
    def __init__(self):
        self.env_backend = EnvironmentCpp(0.01)
        self.initial_positions = np.array([
            [0.5, 0.5, 0],
            [-0.5, 0.5, 0],
            [-0.5, -0.5, 0],
            [0.5, -0.5, 0],
        ])
        for bead_id in range(len(self.initial_positions)):
            self.env_backend.add_bead(
                Bead(bead_id, self.initial_positions[bead_id], mass=1.0, is_mobile=True)
            )
        
        distance_bonds = [
            DistanceBond(0, 1, k=1000),
            DistanceBond(1, 2, k=1000),
            DistanceBond(2, 3, k=1000),
            DistanceBond(3, 0, k=1000),
            DistanceBond(0, 2, k=2000, r0=math.sqrt(2)),
            DistanceBond(1, 3, k=2000, r0=math.sqrt(2)),
        ]
        
        for dist_bond in distance_bonds:
            self.env_backend.add_bond(dist_bond)
        
        rotation_axis = np.array([0, 0, 1.])
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = R.from_rotvec(np.pi * rotation_axis)
        self.target_positions = rotation_matrix.apply(self.initial_positions) + np.array([1, 1, 2])
        self.reward_bottom = np.linalg.norm(self.target_positions - self.initial_positions, axis=1).sum()
        reward_calculator = StayCloseReward({
            bead_id: self.target_positions[bead_id]
            for bead_id in range(len(self.initial_positions))
        })
        self.env_backend.add_reward_calculator(reward_calculator)
        self.count = 0
        
        self.videos = []
        margin = 2
        self.global_mins = np.min([self.target_positions.min(axis=0), self.initial_positions.min(axis=0)], axis=0) - margin
        self.global_maxes = np.max([self.target_positions.max(axis=0), self.initial_positions.max(axis=0)], axis=0) + margin
        
    def reset(self):
        self.env_backend.reset()
        self.count = 0
        self.videos.append([])
        return self._state()
    
    def step(self, action):
        action = {
            0: action[:3],
            1: action[3:6],
            2: action[6:9],
            3: action[9:],    
        }
        partial_rewards = self.env_backend.step(action)
        reward = self.reward_bottom + sum(partial_rewards)
        new_state = self._state()
        self.count += 1
        truncated = (self.count == 1000)
        if truncated:
            info = {"TimeLimit.truncated": truncated}
        else:
            info = {}
        return new_state, reward, (truncated or reward <= -2 * self.reward_bottom), info
    
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
            positions = [bead.get_position() for bead in self.env_backend.get_beads()]
            positions_closed = np.r_[positions + positions[:1]]
            x, y, z = positions_closed.T
            positions = np.r_[positions]
            # TODO: make this automatic based on the bonds
            ax0.plot(x, y, z, "b", linewidth=3, label="bonds")
            ax0.scatter(positions[:, 0], positions[:, 1], positions[:, 2], linewidth=10, label="beads")
            ax0.scatter(self.target_positions[:, 0], self.target_positions[:, 1], self.target_positions[:, 2], linewidth=20, alpha=0.3)
            ax0.set_xlim(self.global_mins[0], self.global_maxes[0])
            ax0.set_ylim(self.global_mins[1], self.global_maxes[1])
            ax0.set_zlim(self.global_mins[2], self.global_maxes[2])
            
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
            # np.linalg.norm(vectorized[:3] - vectorized[9:12]),
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
        return Box(low=self.reward_bottom, high=16.0, shape=(1,), dtype=np.float32)
        
    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(len(self._state()),), dtype=np.float32)
    
    @property
    def action_space(self):
        low = np.array(12 * [-5], dtype=np.float32)
        high = np.array(4 * [5, 5, 20], dtype=np.float32)
        return Box(low=low, high=high, shape=(4 * 3,), dtype=np.float32)

    def seed(self, seed=None):
        pass


if __name__ == "__main__":
    env = BeadsQuadCopterEnvironment()
