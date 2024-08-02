import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
plt.switch_backend('TkAgg')

import hydra
from omegaconf import DictConfig, OmegaConf

from beads_gym.environment.environment import Environment
from beads_gym.beads.beads import Bead
from beads_gym.bonds.bonds import DistanceBond
from beads_gym.environment.reward.reward import Reward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()



@hydra.main(config_path="conf", config_name="dupa")
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
    x, y, z = positions.T
    # scatter = ax0.scatter(x, y, z, 'b', linewidth=10, label='reference')
    plot, = ax0.plot(x, y, z, 'b', linewidth=1, label='bonds')
    scatter = ax0.scatter([], [], [], 'b', linewidth=10, label='beads')
    scatter3d = ax0.scatter3D([], [], [], 'b', linewidth=10, label='beads')
    from mpl_toolkits.mplot3d.art3d import Path3DCollection, Line3D
    
    def update(num):
        beads = env.get_beads()
        positions = np.r_[[bead.get_position() for bead in beads]]
        x, y, z = positions.T
        plot.set_xdata(x)
        plot.set_ydata(y)
        plot.set_3d_properties(z)
        scatter.set_offsets(positions[:, :2])
        scatter.set_3d_properties(positions[:, 2], zdir='z')
        env.step({0: np.random.normal(size=3), 1: np.random.normal(size=3)})
        return scatter, plot
    
    ani = animation.FuncAnimation(fig, update, frames=500, interval=10, repeat=False, blit=True)
    ani.save("a.mp4", writer="ffmpeg")
    # plt.show()


if __name__ == "__main__":
    main()
    
    env = Environment()
    
    bead_1 = Bead(0, [0, 0, 0], 1.0, False)
    bead_2 = Bead(1, [0, 0, 1], 1.0, True)
    env.add_bead(bead_1)
    env.add_bead(bead_2)
    
    distance_bond = DistanceBond(0, 1)
    env.add_bond(distance_bond)
    
    reference_beads = [
        Bead(2, [-1, -1, -1], 1.0, False),
        Bead(3, [2, 2, 2], 1.0, False),
    ]
    reward = Reward(reference_beads)
    env.add_reward(reward)

    print(f'env.get_beads() = {env.get_beads()}')
    print(f'env.get_bonds()[0].get_velocity() = {env.get_beads()[0].get_velocity()}')
    
    try_out_animation(env)
