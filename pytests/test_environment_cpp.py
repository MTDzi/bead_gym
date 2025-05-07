import numpy as np
import pytest
from time import perf_counter

from beads_gym.environment.environment_cpp import EnvironmentCpp
from beads_gym.beads.beads import Bead
from beads_gym.bonds.bonds import DistanceBond
from beads_gym.environment.reward.rewards import StayCloseReward


@pytest.fixture
def setup_env():
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
    })
    env.add_reward_calculator(reward)
    return env


def test_beads_and_bonds_initialization(setup_env):
    env = setup_env
    beads = env.get_beads()
    assert len(beads) == 2
    assert beads[0].get_position()[2] == pytest.approx(0.0)
    assert beads[1].get_position()[2] == pytest.approx(0.9)

    assert len(env.get_beads()) == 2


def test_step_performance(setup_env):
    env = setup_env
    one_action = {0: np.random.normal(size=3), 1: np.random.normal(size=3)}
    num_steps = 10000
    start = perf_counter()
    for _ in range(num_steps):
        env.step(one_action)
    elapsed_us = 1_000_000 * (perf_counter() - start) / num_steps
    print(f"Avg step time: {elapsed_us:.2f} µs")
    assert elapsed_us < 500.0  # adjust depending on expected perf


def test_reset_functionality(setup_env):
    env = setup_env
    one_action = {0: np.random.normal(size=3), 1: np.random.normal(size=3)}
    for _ in range(100):
        env.step(one_action)
    env.reset()
    beads = env.get_beads()
    # Assume reset puts them back at initial position
    assert np.allclose(beads[0].get_position(), [0, 0, 0], atol=1e-2)
    assert np.allclose(beads[1].get_position(), [0, 0, 0.9], atol=1e-2)
