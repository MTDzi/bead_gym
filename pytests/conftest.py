import pytest
import numpy as np

from beads_gym.environment.environment_cpp import EnvironmentCpp
from beads_gym.beads.beads import Bead
from beads_gym.bonds.bonds import DistanceBond
from beads_gym.environment.reward.rewards import StayCloseReward


@pytest.fixture
def env_cpp():
    env = EnvironmentCpp(0.001, 10, 0.99, 0.00001)

    bead_0 = Bead(0, [0, 0, 0], 1.0, True, [2])
    bead_1 = Bead(1, [0, 0, 0.9], 1.0, True)
    env.add_bead(bead_0)
    env.add_bead(bead_1)

    bond = DistanceBond(0, 1)
    env.add_bond(bond)

    reward = StayCloseReward({
        0: np.array([0, 0, 0]),
        1: np.array([0, 0, 1]),
    }, [1, 1])
    env.add_reward_calculator(reward)
    return env
