import time
import warnings
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv, VectorEnv


class Collector:
    def __init__(
        self,
        policy: None,
        env: gym.Env | VectorEnv,
        buffer: None = None,
        exploration_noise: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(env, gym.Env) and not hasattr(env, "env_fns"):
            warnings.warn("Single environment detected, wrap to SyncVectorEnv.")
            self.env = SyncVectorEnv([lambda: env])
        else:
            self.env = env  # type: ignore

        self.env_num = self.env.num_envs
        self.exploration_noise = exploration_noise
        self.buffer = buffer
        self.policy = policy

        self.collect_step: int = 0
        self.collect_episode: int = 0
        self.collect_time: float = 0.0

        self._action_space = self.env.action_space
        self._pre_obs: np.ndarray | None = None
        self._pre_info: dict | None = None
        self._is_closed: bool = False

    def close(self) -> None:
        """Close the collector and the environment."""
        self.env.close()
        self._pre_obs = None
        self._pre_info = None
        self._is_closed = True

    @property
    def is_closed(self) -> bool:
        """Return True if the collector is closed."""
        return self._is_closed

    def reset(
        self,
        reset_buffer: bool = True,
        reset_stats: bool = True,
        seed: int | None = None,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environment, statistics, and data needed to start the collection."""
        self.reset_env(seed=seed, gym_reset_kwargs=gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        if reset_stats:
            self.reset_stat()
        self._is_closed = False

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        # TODO: reset buffer

    def reset_env(
        self,
        seed: int | None = None,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environments and the initial obs, info, and hidden state of the collector."""
        gym_reset_kwargs = gym_reset_kwargs or {}

        if seed is not None and "seed" in gym_reset_kwargs:
            raise ValueError("Cannot specify both 'seed' and 'gym_reset_kwargs['seed']'.")

        if seed is not None:
            gym_reset_kwargs["seed"] = seed

        self._pre_obs, self._pre_info = self.env.reset(**gym_reset_kwargs)

    def collect(
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: bool = False,
        no_grad: bool = True,
        reset_before_collect: bool = False,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if (n_step is not None and n_episode is not None) or (n_step is None and n_episode is None):
            raise ValueError(
                f"Either n_step or n_episode must be specified (but not both or none), but got {n_step=}, {n_episode=}.",
            )
        if n_step is not None:
            assert n_step > 0, f"n_step must be positive, but got {n_step=}."
        if n_episode is not None:
            assert n_episode > 0, f"n_episode must be positive, but got {n_episode=}."

        if n_episode is not None and n_episode < self.env_num:
            warnings.warn(
                "You are trying to collect fewer episodes than the number of environments. "
                f"Got {n_episode=}, while the number of environments is {self.env_num}.",
            )

        start_time = time.time()

        if reset_before_collect:
            self.reset(reset_buffer=False, gym_reset_kwargs=gym_reset_kwargs)

        # TODO: implement
        # https://raw.githubusercontent.com/aai-institute/tianshou/master/tianshou/data/collector.py
        step_count = 1
        num_collected_episodes = 1

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += num_collected_episodes
        collect_time = max(time.time() - start_time, 1e-9)
        self.collect_time += collect_time
