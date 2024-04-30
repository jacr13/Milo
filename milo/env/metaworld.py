import random
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import metaworld
from gymnasium import Env, Wrapper

from milo.env.utils import gym_vector_env_creator


def make(
    env_id: str,
    num_envs: int = 1,
    vectorization_mode: str = "async",
    env_spec_kwargs: dict[str, Any] | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    wrappers: Sequence[Callable[[Env], Wrapper]] | None = None,
) -> gym.vector.VectorEnv:
    """Creates a metaworld vectorized environment based on the given environment ID, number of environments, vectorization mode, environment specific arguments, vector arguments, and wrappers."""
    env_spec_kwargs = env_spec_kwargs or {}
    vector_kwargs = vector_kwargs or {}
    wrappers = wrappers or []
    render_mode = env_spec_kwargs.pop("render_mode", "rgb_array")

    ml1 = metaworld.ML1(env_id)
    env_fns = [lambda: ml1.train_classes[env_id](render_mode=render_mode, **env_spec_kwargs) for _ in range(num_envs)]

    envs = gym_vector_env_creator(env_fns, vectorization_mode, vector_kwargs=vector_kwargs)

    # Randomly sample tasks, assumes num_env <= len(train_tasks)
    assert num_envs <= len(ml1.train_tasks), "num_envs must be <= len(train_tasks)"

    tasks = random.sample(ml1.train_tasks, num_envs)
    envs.call_ids("set_task", [{"args": [task]} for task in tasks])

    return envs
