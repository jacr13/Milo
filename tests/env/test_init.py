import copy

import gymnasium as gym
import numpy as np
import pytest

from milo.env import find_simulator, make_env


def test_find_simulator():
    """Tests for the find_simulator function."""
    # Test with an unsupported environment ID
    assert find_simulator("UnsupportedEnv-v0") is None

    # Test with an environment ID from gymnasium
    assert find_simulator("CartPole-v1") == "gymnasium"

    # Test with an environment ID from metaworld (need metaworld installed)
    try:
        import metaworld  # noqa: F401

        assert find_simulator("button-press-topdown-v2") == "metaworld"
    except ModuleNotFoundError:
        assert find_simulator("button-press-topdown-v2") is None

    # Test with an environment ID from dmc (need dmc installed)
    try:
        import dm_control  # noqa: F401

        assert find_simulator("walker-walk") == "dmc"
    except ModuleNotFoundError:
        assert find_simulator("walker-walk") is None


env_configs = [
    # gymnasium
    {
        "num_envs": 1,
        "vectorization_mode": "async",
        "env_type": gym.vector.AsyncVectorEnv,
    },
    {
        "num_envs": 2,
        "vectorization_mode": "async",
        "env_type": gym.vector.AsyncVectorEnv,
    },
    {
        "num_envs": 1,
        "vectorization_mode": "sync",
        "env_type": gym.vector.SyncVectorEnv,
    },
    {
        "num_envs": 2,
        "vectorization_mode": "sync",
        "env_type": gym.vector.SyncVectorEnv,
    },
]


class TestMakeEnv:
    """Tests for the make_env function."""
    @pytest.mark.parametrize("env_config", env_configs)
    def test_create_environment(self, env_config):
        for env_id in ["CartPole-v1", "button-press-topdown-v2", "walker-walk"]:
            env_config_copy = copy.deepcopy(env_config)
            env_type = env_config_copy.pop("env_type")

            env = make_env(env_id, **env_config_copy)

            assert isinstance(env, gym.vector.VectorEnv)
            assert isinstance(env, env_type)
            assert env.num_envs == env_config_copy["num_envs"]

            reset_result = env.reset()
            assert len(reset_result) == 2
            assert reset_result[0].shape == env.observation_space.shape

    @pytest.mark.parametrize("env_config", env_configs)
    def test_reset_seed_step_environment(self, env_config):
        for env_id in ["CartPole-v1", "button-press-topdown-v2", "walker-walk"]:
            env_config_copy = copy.deepcopy(env_config)
            env_type = env_config_copy.pop("env_type")
            num_envs = env_config_copy["num_envs"]
        
            env = make_env(env_id, **env_config_copy)

            reset_result = env.reset(seed=13)
            assert len(reset_result) == 2
            assert reset_result[0].shape == env.observation_space.shape

            action = env.action_space.sample()
            assert action.shape == env.action_space.shape

            obs, rew, terminated, truncated, info = env.step(action)

            assert obs.shape == env.observation_space.shape
            assert not np.all(obs == reset_result[0])
            assert rew.shape == (num_envs,)
            assert terminated.shape == (num_envs,)
            assert truncated.shape == (num_envs,)
            assert isinstance(info, dict)

            reset_result_2 = env.reset(seed=13)
            assert len(reset_result_2) == 2
            assert reset_result_2[0].shape == env.observation_space.shape
            assert np.all(reset_result[0] == reset_result_2[0])

    def test_raise_value_error_no_simulator(self):
        env_id = "UnsupportedEnv-v1"

        with pytest.raises(ValueError):
            make_env(env_id)

    def test_raise_value_error_no_module_simulator(self):
        env_id = "UnsupportedEnv-v1"
        simulator = "UnsupportedSimulator"

        with pytest.raises(ValueError):
            make_env(env_id, simulator=simulator)