from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
from gymnasium import Env, Wrapper


def make(
    env_id: str,
    num_envs: int = 1,
    vectorization_mode: str = "async",
    env_spec_kwargs: dict[str, Any] | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    wrappers: Sequence[Callable[[Env], Wrapper]] | None = None,
) -> gym.vector.VectorEnv:
    """TODO: Fill."""
    if env_spec_kwargs is None:
        env_spec_kwargs = {}
    if vector_kwargs is None:
        vector_kwargs = {}
    if wrappers is None:
        wrappers = []

    return gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        vector_kwargs=vector_kwargs,
        wrappers=wrappers,
        **env_spec_kwargs,
    )
