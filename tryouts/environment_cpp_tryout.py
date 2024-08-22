import argparse
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
plt.switch_backend('TkAgg')

import hydra
from omegaconf import DictConfig, OmegaConf

from beads_gym.environment.environment_cpp import EnvironmentCpp
from beads_gym.beads.beads import Bead
from beads_gym.bonds.bonds import DistanceBond
from beads_gym.environment.reward.rewards import StayCloseReward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()



@hydra.main(config_path="conf", config_name="environment_cpp_tryout")
def main(cfg: DictConfig) -> None:
    bead_cfg_dict = OmegaConf.to_container(cfg.beads[0], resolve=True)
    some_initialized_bead = hydra.utils.instantiate(bead_cfg_dict)
    print(cfg)


def try_out_animation(env):
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
    positions = np.r_[[bead.get_position() for bead in env.get_beads()]]
    positions_closed = np.r_[positions, positions[0]]
    x, y, z = positions_closed.T
    plot, = ax0.plot(x, y, z, 'b', linewidth=1, label='bonds')
    scatter = ax0.scatter([], [], [], 'b', linewidth=10, label='beads')
    ax0.set_xlim(-0.5, 0.5)
    ax0.set_ylim(-0.5, 0.5)
    ax0.set_zlim(0, 1.5)
    
    def update(num):
        beads = env.get_beads()
        positions = np.r_[[bead.get_position() for bead in beads]]
        x, y, z = positions.T
        plot.set_xdata(x)
        plot.set_ydata(y)
        plot.set_3d_properties(z)
        scatter.set_offsets(positions[:, :2])
        scatter.set_3d_properties(positions[:, 2], zdir='z')
        partial_rewards = env.step({0: np.random.normal(size=3), 1: np.random.normal(size=3)})
        # print(f'bead_1.position: {beads[1].get_position()}')
        # print(f'reward = {sum(partial_rewards)}, potential = {env.calc_bond_potential()}')
        return scatter, plot
    
    ani = animation.FuncAnimation(fig, update, frames=300, interval=5, repeat=False, blit=True)
    ani.save("a.mp4", writer="ffmpeg")


if __name__ == "__main__":
    main()
    
    env = EnvironmentCpp(0.01)
    
    bead_0 = Bead(0, [0, 0, 0], 1.0, True, [2])
    bead_1 = Bead(1, [0, 0, 0.9], 1.0, True)
    env.add_bead(bead_0)
    env.add_bead(bead_1)
    
    distance_bond = DistanceBond(0, 1)
    env.add_bond(distance_bond)
    
    reward_calculator = StayCloseReward({
        0: np.array([0, 0, 0]),
        1: np.array([0, 0, 1]),
    })

    env.add_reward_calculator(reward_calculator)

    print(f'env.get_beads() = {env.get_beads()}')
    print(f'env.get_bonds()[0].get_velocity() = {env.get_beads()[0].get_velocity()}')

    num_steps = 10000
    one_action = {0: np.random.normal(size=3), 1: np.random.normal(size=3)}
    start = perf_counter()
    for i in range(num_steps):
        env.step(one_action)
    print(f'One steps took on average: {(1_000_000 * (perf_counter() - start) / num_steps):.2f} [us]')
    env.reset()
    
    one_action = {0: np.random.normal(size=3), 1: np.random.normal(size=3)}
    start = perf_counter()
    for i in range(num_steps):
        env.step(one_action)
    print(f'One steps took on average: {(1_000_000 * (perf_counter() - start) / num_steps):.2f} [us]')
    env.reset()

    try_out_animation(env)
    one_action = {0: np.random.normal(size=3), 1: np.random.normal(size=3)}
    start = perf_counter()
    for i in range(num_steps):
        env.step(one_action)
    print(f'One steps took on average: {(1_000_000 * (perf_counter() - start) / num_steps):.2f} [us]')
    env.reset()
    
    one_action = {0: np.random.normal(size=3), 1: np.random.normal(size=3)}
    start = perf_counter()
    for i in range(num_steps):
        env.step(one_action)
    print(f'One steps took on average: {(1_000_000 * (perf_counter() - start) / num_steps):.2f} [us]')
    env.reset()
