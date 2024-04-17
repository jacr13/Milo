import importlib
import os
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
from gymnasium import Env, Wrapper


def make_env(
    env_id: str,
    num_envs: int = 1,
    vectorization_mode: str = "async",
    simulator: str | None = None,
    env_spec_kwargs: dict[str, Any] | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    wrappers: Sequence[Callable[[Env], Wrapper]] | None = None,
) -> gym.vector.VectorEnv:
    """Creates a vectorized environment based on the provided parameters."""
    if env_spec_kwargs is None:
        env_spec_kwargs = {}
    if vector_kwargs is None:
        vector_kwargs = {}
    if wrappers is None:
        wrappers = []

    # f simulator is not provided, attempt to detect simulator automatically
    if simulator is None:
        simulator = find_simulator(env_id)

    # if simulator is still None raise error
    if simulator is None:
        raise ValueError(f"Cannot create environment {env_id} because we couldn't find a simulator to use.")

    # if simulator found or given create environment
    try:
        make = importlib.import_module(f"milo.env.{simulator}").make

        return make(
            env_id,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            env_spec_kwargs=env_spec_kwargs,
            vector_kwargs=vector_kwargs,
            wrappers=wrappers,
        )
    except ModuleNotFoundError as exc:
        raise ValueError(f"Cannot create environment {env_id} because simulator {simulator} is not supported.") from exc


def find_simulator(env_id: str) -> str | None:
    """Checks for the existence of an environment ID in various simulation libraries and returns the name of the library if the ID is found. If the ID is not found in any of the libraries, it returns None."""
    # gymnasium
    try:
        import gymnasium as gym

        if env_id in gym.envs.registry:
            return "gymnasium"
    except ModuleNotFoundError:
        pass

    # metaworld
    try:
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

        if env_id in ALL_V2_ENVIRONMENTS:
            return "metaworld"
    except ModuleNotFoundError:
        pass

    # dmc
    try:
        # dmc with headless rendering to avoid warning
        os.environ["MUJOCO_GL"] = "osmesa"
        from dm_control.suite import ALL_TASKS

        all_tasks = ["-".join(task) for task in ALL_TASKS]
        if env_id in all_tasks:
            return "dmc"
    except ModuleNotFoundError:
        pass

    return None
