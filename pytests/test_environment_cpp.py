import numpy as np
import pytest
from time import perf_counter


def test_beads_and_bonds_initialization(env_cpp):
    beads = env_cpp.get_beads()
    assert len(beads) == 2
    assert beads[0].get_position()[2] == pytest.approx(0.0)
    assert beads[1].get_position()[2] == pytest.approx(0.9)

    assert len(env_cpp.get_beads()) == 2


def test_step_performance(env_cpp):
    one_action = {0: np.random.normal(size=3), 1: np.random.normal(size=3)}
    num_steps = 10000
    start = perf_counter()
    for _ in range(num_steps):
        env_cpp.step(one_action)
    elapsed_us = 1_000_000 * (perf_counter() - start) / num_steps
    print(f"Avg step time: {elapsed_us:.2f} µs")
    assert elapsed_us < 500.0  # adjust depending on expected perf


def test_reset_functionality(env_cpp):
    one_action = {0: np.random.normal(size=3), 1: np.random.normal(size=3)}
    for _ in range(100):
        env_cpp.step(one_action)
    env_cpp.reset()
    beads = env_cpp.get_beads()
    # Assume reset puts them back at initial position
    assert np.allclose(beads[0].get_position(), [0, 0, 0], atol=1e-2)
    assert np.allclose(beads[1].get_position(), [0, 0, 0.9], atol=1e-2)
