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
    """Creates a gym vectorized environment based on the given environment ID, number of environments, vectorization mode, environment specific arguments, vector arguments, and wrappers."""
    env_spec_kwargs = env_spec_kwargs or {}
    vector_kwargs = vector_kwargs or {}
    wrappers = wrappers or []

    # Set default render mode
    if "render_mode" not in env_spec_kwargs:
        env_spec_kwargs["render_mode"] = "rgb_array"

    return gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        vector_kwargs=vector_kwargs,
        wrappers=wrappers,
        **env_spec_kwargs,
    )
