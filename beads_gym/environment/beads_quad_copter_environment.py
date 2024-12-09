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
        self.env_backend = EnvironmentCpp(0.001, 10, 1.0, 0.000)
        self.initial_positions = np.array([
            [0.5, 0.5, 0],
            [-0.5, 0.5, 0],
            [-0.5, -0.5, 0],
            [0.5, -0.5, 0],
            
            # Extra Beads to make it stiffer
            [0, 0, -0.25],
            
            # And the bead on top
            [0, 0, 1.25],
        ])
        for bead_id in range(len(self.initial_positions)):
            self.env_backend.add_bead(
                Bead(bead_id, self.initial_positions[bead_id], mass=1.0, is_mobile=True)
            )
            
        self.bonds = [
            (0, 1, 3000),
            (1, 2, 3000),
            (2, 3, 3000),
            (3, 0, 3000),
            (0, 2, 4000),
            (1, 3, 4000),
            
            (0, 4, 4000),
            (1, 4, 4000),
            (2, 4, 4000),
            (3, 4, 4000),
            
            (4, 5, 4000)
        ]
        
        distance_bonds = [
            DistanceBond(i, j, k=k, r0=np.linalg.norm(self.initial_positions[i] - self.initial_positions[j]))
            for i, j, k in self.bonds
        ]
        
        for dist_bond in distance_bonds:
            self.env_backend.add_bond(dist_bond)
        
        translation_vec = np.array([0.0, 0., 0.])
        rotation_angle = 0 #  np.pi
        rotation_axis = np.array([0, 0, 1.])
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = R.from_rotvec(rotation_angle * rotation_axis)
        self.target_positions = rotation_matrix.apply(self.initial_positions) + translation_vec
        self.reward_bottom = np.linalg.norm(self.target_positions - self.initial_positions, axis=1).sum()
        if self.reward_bottom < 0.1:
            self.reward_bottom = len(self.target_positions)  # TODO: IDK, could be anything
        reward_calculator = StayCloseReward({
            bead_id: self.target_positions[bead_id]
            for bead_id in range(len(self.initial_positions))
        })
        self.env_backend.add_reward_calculator(reward_calculator)
        bead_on_top_id = len(self.initial_positions) - 1
        self.env_backend.add_reward_calculator(
            StayCloseReward({
                bead_on_top_id: self.target_positions[bead_on_top_id]
            })
        )
        self.env_backend.add_reward_calculator(
            StayCloseReward({
                bead_on_top_id: self.target_positions[bead_on_top_id]
            })
        )
        self.env_backend.add_reward_calculator(
            StayCloseReward({
                bead_on_top_id: self.target_positions[bead_on_top_id]
            })
        )
        self.count = 0
        
        self.videos = []
        margin = 1
        self.global_mins = np.min([self.target_positions.min(axis=0), self.initial_positions.min(axis=0)], axis=0) - margin
        self.global_maxes = np.max([self.target_positions.max(axis=0), self.initial_positions.max(axis=0)], axis=0) + margin
        
    def reset(self):
        self.env_backend.reset()
        self.count = 0
        self.videos = []
        self.videos.append([])
        return self._state()
    
    def step(self, action):
        anti_gravity = np.array([0.0, 0.0, 9.81])
        action = {
            0: action[:3] + anti_gravity,
            1: action[3:6] + anti_gravity,
            2: action[6:9] + anti_gravity,
            3: action[9:] + anti_gravity,    
        }
        partial_rewards = self.env_backend.step(action)
        reward = self.reward_bottom + sum(partial_rewards)
        new_state = self._state()
        self.count += 1
        truncated = (self.count == 2500)
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
            for i, j, _ in self.bonds:
                ax0.plot(positions[[i, j], 0], positions[[i, j], 1], positions[[i, j], 2], "b", linewidth=3)
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
            
            if self.count % 50 == 0:
                self.videos[-1].append(rgb_array)
            
            return rgb_array

    def _state(self):
        beads = self.env_backend.get_beads()
        vectorized = np.r_[
            # [[bead.get_position(), bead.get_velocity(), bead.get_acceleration(), bead.get_external_acceleration() / 100] for bead in beads]
            [[bead.get_position(), bead.get_velocity(), bead.get_acceleration()] for bead in beads]
        ].flatten()
        # print(np.r_[
        #     [[bead.get_external_acceleration()] for bead in beads]
        # ].flatten())
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
        return Box(low=-2 * self.reward_bottom, high=self.reward_bottom, shape=(1,), dtype=np.float32)
        
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
    env.reset()
    plt.imshow(env.render())
    plt.savefig("test.png")
    
