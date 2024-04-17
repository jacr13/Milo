import pytest
import numpy as np
from gymnasium.spaces import Box

from milo.env.dmc import DMC2Gym


@pytest.fixture
def dmc_env():
    return DMC2Gym(domain="walker", task="run")

def test_initialization(dmc_env):
    assert isinstance(dmc_env.observation_space, Box)
    assert isinstance(dmc_env.action_space, Box)
    assert dmc_env.reward_range == (0, 1)

def test_step(dmc_env):
    action = np.zeros(dmc_env.action_space.shape)
    observation, reward, terminated, truncated, info = dmc_env.step(action)
    assert isinstance(observation, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_reset(dmc_env):
    observation, info = dmc_env.reset()
    assert isinstance(observation, np.ndarray)
    assert isinstance(info, dict)

def test_seed_reset(dmc_env):
    initial_observation, _ = dmc_env.reset(seed=12)
    dmc_env.step(dmc_env.action_space.sample())
    new_observation, _ = dmc_env.reset(seed=12)
    assert np.array_equal(initial_observation, new_observation)

def test_render(dmc_env):
    render_frame = dmc_env.render()
    assert isinstance(render_frame, np.ndarray)
    assert render_frame.shape == (dmc_env.render_height, dmc_env.render_width, 3)

def test_render_size(dmc_env):
    render_frame = dmc_env.render(height=100, width=200)
    assert render_frame.shape == (100, 200, 3)

def test_getattr(dmc_env):
    assert hasattr(dmc_env, "reset")
    assert hasattr(dmc_env, "step")
    assert hasattr(dmc_env, "render")

def test_invalid_action(dmc_env):
    action = np.ones(dmc_env.action_space.shape) * 10
    with pytest.raises(AssertionError):
        dmc_env.step(action)

def test_invalid_render_mode():
    with pytest.raises(AssertionError):
        DMC2Gym(domain="cartpole", task="swingup", rendering="invalid_mode")

def test_seed_task_kwargs_action_space():
    env = DMC2Gym(domain="cartpole", task="swingup", task_kwargs={"random": 13})
    actions_13 = env.action_space.sample()

    env = DMC2Gym(domain="cartpole", task="swingup", task_kwargs={"random": 99})
    actions_99 = env.action_space.sample()

    env = DMC2Gym(domain="cartpole", task="swingup", task_kwargs={"random": 13})
    actions_13_bis = env.action_space.sample()

    assert not np.array_equal(actions_13, actions_99)
    assert np.array_equal(actions_13, actions_13_bis)

def test_seed_task_kwargs_observation_space():
    env = DMC2Gym(domain="cartpole", task="swingup", task_kwargs={"random": 13})
    obs_13 = env.observation_space.sample()

    env = DMC2Gym(domain="cartpole", task="swingup", task_kwargs={"random": 99})
    obs_99 = env.observation_space.sample()

    env = DMC2Gym(domain="cartpole", task="swingup", task_kwargs={"random": 13})
    obs_13_bis = env.observation_space.sample()

    assert not np.array_equal(obs_13, obs_99)
    assert np.array_equal(obs_13, obs_13_bis)