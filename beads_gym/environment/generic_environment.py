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


class GenericEnvironment:
    def __init__(
        self,
        env_backend: EnvironmentCpp,
        beads: list[Bead],
        bonds: list[DistanceBond],
        which_beads_actuate: list[int],
        do_render: bool | None,
        do_record: bool | None,
        monitor_mode: bool | None,
        max_num_steps: int = 2500,
    ):
        self.env_backend = env_backend
        
        self.do_render = do_render
        self.do_record = do_record
        self.monitor_mode = monitor_mode
        self.max_num_steps = max_num_steps
                
        for bead in beads:
            self.env_backend.add_bead(bead)
        self.which_beads_actuate = which_beads_actuate
        
        for bond in bonds:
            self.env_backend.add_bond(bond)
        self.bonds = [
            bond.get_bead_ids() for bond in bonds
        ]
            
        # TODO: Configuring rewards in a .yaml file is, it turns out,
        #  difficult, if we wanted to pass the reward(s) as __init__
        #  arguments. For now, I'm hard-coding the StayCloseReward reward
        #  and once more reward types emerge, I'll deal with this more
        #  elegantly.
        # reward_calculator = StayCloseReward({
        #     bead_id: self.target_positions[bead_id]
        #     for bead_id in range(len(self.initial_positions))
        # })
        # self.env_backend.add_reward_calculator(reward_calculator)
        # bead_on_top_id = len(self.initial_positions) - 1
        # self.env_backend.add_reward_calculator(
        #     StayCloseReward({
        #         bead_on_top_id: self.target_positions[bead_on_top_id]
        #     })
        # )
        # self.env_backend.add_reward_calculator(
        #     StayCloseReward({
        #         bead_on_top_id: self.target_positions[bead_on_top_id]
        #     })
        # )
        # self.env_backend.add_reward_calculator(
        #     StayCloseReward({
        #         bead_on_top_id: self.target_positions[bead_on_top_id]
        #     })
        # )
        weights = (len(beads) - 1) * [1] + [4]
        self.env_backend.add_reward_calculator(StayCloseReward.from_beads(beads, weights))
        self.reward_bottom = len(beads)  # TODO: IDK, can be anything
        
        self.step_count = 0
        self.videos = []
        margin = 1
        initial_positions = np.array([bead.get_position() for bead in beads])
        self.global_mins = initial_positions.min(axis=0) - margin
        self.global_maxes = initial_positions.max(axis=0) + margin
        
    def reset(self):
        self.env_backend.reset()
        self.step_count = 0
        self.videos = [[]]
        return self._state()
            
    def step(self, action):
        anti_gravity = np.array([0.0, 0.0, 9.81])
        action = {
            bead_id: action[(3 * i):(3 * i + 3)] + anti_gravity
            for i, bead_id in enumerate(self.which_beads_actuate)
        }
        partial_reward = self.env_backend.step(action)
        reward = self.reward_bottom + sum(partial_reward)
        new_state = self._state()
        self.step_count += 1
        truncated = (self.step_count == self.max_num_steps)
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
            for i, j in self.bonds:
                ax0.plot(positions[[i, j], 0], positions[[i, j], 1], positions[[i, j], 2], "b", linewidth=3)
            ax0.scatter(positions[:, 0], positions[:, 1], positions[:, 2], linewidth=10, label="beads")
            ax0.set_xlim(self.global_mins[0], self.global_maxes[0])
            ax0.set_ylim(self.global_mins[1], self.global_maxes[1])
            ax0.set_zlim(self.global_mins[2], self.global_maxes[2])
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            
            img = Image.open(buf).convert("RGB")
            rgb_array = np.array(img)
            plt.close(fig)
            
            if self.step_count % 50 == 0:
                self.videos[-1].append(rgb_array)
            
            return rgb_array
        
    def _state(self):
        beads = self.env_backend.get_beads()
        vectorized = np.r_[
            [[bead.get_position(), bead.get_velocity(), bead.get_acceleration(), bead.get_external_acceleration() / 100.] for bead in beads]
            # [[bead.get_position(), bead.get_velocity(), bead.get_acceleration()] for bead in beads]
        ].flatten()
        # print(np.r_[
        #     [[bead.get_external_acceleration()] for bead in beads]
        # ].max())
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
        bound = 6
        low = np.array(12 * [-bound], dtype=np.float32)
        high = np.array(12 * [bound], dtype=np.float32)
        return Box(low=low, high=high, shape=(4 * 3,), dtype=np.float32)

    def seed(self, seed=None):
        pass
