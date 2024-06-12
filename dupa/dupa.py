import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
plt.switch_backend('TkAgg')


import hydra
from omegaconf import DictConfig, OmegaConf

from beads_gym.environment.environment import Environment
from beads_gym.beads.beads import Bead, ThreeDegreesOfFreedomBead
from beads_gym.bonds.bonds import DistanceBond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()



@hydra.main(config_path="conf", config_name="dupa")
def main(cfg: DictConfig) -> None:
    bead_cfg_dict = OmegaConf.to_container(cfg.beads[0], resolve=True)
    some_initialized_bead = hydra.utils.instantiate(bead_cfg_dict)
    print(cfg)


def try_out_animation():
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
    x = np.random.randn(1)
    y = np.random.randn(1)
    z = np.random.randn(1)
    cloud = ax0.plot(x, y, z, 'b', linewidth=1, label='reference')
    
    def update(num):
        x = np.random.randn(100)
        y = np.random.randn(100)
        z = np.random.randn(100)
        cloud = ax0.plot(x, y, z, 'b', linewidth=1, label='reference')
        return cloud
    
    ani = animation.FuncAnimation(fig, update, frames=20000, interval=1, repeat=True, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
    bead = Bead(0, [1, 1, 1])
    tdf_bead = ThreeDegreesOfFreedomBead(1, [1, 1, 1], 1.0)
    
    distance_bond = DistanceBond(0, 1)

    env = Environment()
    env.add_bead(bead)
    env.add_bead(tdf_bead)
    env.add_bond(distance_bond)

    print(f'env.get_beads() = {env.get_beads()}')
    
    # try_out_animation()