import warnings
from typing import Any

import gymnasium as gym
from gymnasium.vector.sync_vector_env import SyncVectorEnv


class Collector:
    def __init__(
        self,
        policy: None,
        env: gym.Env | gym.vector.VectorEnv,
        buffer: None = None,
        exploration_noise: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(env, gym.Env) and not hasattr(env, "__len__"):
            warnings.warn("Single environment detected, wrap to SyncVectorEnv.")
            self.env = SyncVectorEnv([lambda: env])
        else:
            self.env = env

        self.env_num = len(self.env)
        self.exploration_noise = exploration_noise
        self.buffer = self._assign_buffer(buffer)
        self.policy = policy

        self._action_space = self.env.action_space
        self._is_closed = False

    def close(self) -> None:
        """Close the collector and the environment."""
        self.env.close()
        self._is_closed = True

    @property
    def is_closed(self) -> bool:
        """Return True if the collector is closed."""
        return self._is_closed

    def reset(
        self,
        reset_buffer: bool = True,
        reset_stats: bool = True,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environment, statistics, and data needed to start the collection."""
        self.reset_env(gym_reset_kwargs=gym_reset_kwargs)
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

    def reset_env(
        self,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environments and the initial obs, info, and hidden state of the collector."""
        gym_reset_kwargs = gym_reset_kwargs or {}
        _, _ = self.env.reset(**gym_reset_kwargs)

    # TODO: reduce complexity, remove the noqa
    def collect(
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: float | None = None,
        no_grad: bool = True,
        reset_before_collect: bool = False,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ):
        # TODO: implement
        pass
