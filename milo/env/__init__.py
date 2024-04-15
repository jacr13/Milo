import importlib

import gymnasium as gym


def make_env(env_id: str, num_envs: int = 1, vectorization_mode: str = "async", **kwargs) -> gym.vector.VectorEnv:
    # check if simulator provided
    simulator = kwargs.get("simulator", None)

    # if not attempt to detect simulator automatically
    if simulator is None:
        simulator = find_simulator(env_id)

    # if simulator is still None raise error
    if simulator is None:
        raise ValueError(f"Cannot create environment {env_id} because we couldn't find a simulator to use.")

    # if simulator found or given create environment
    try:
        make = getattr(importlib.import_module(f"milo.env.{simulator}"), "make")

        return make(
            env_id,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            **kwargs,
        )
    except ModuleNotFoundError:
        raise ValueError(f"Cannot create environment {env_id} because simulator {simulator} is not supported.")


def find_simulator(env_id: str) -> str | None:
    # gymnasium
    try:
        import gymnasium as gym

        if env_id in gym.envs.registry.keys():
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
        from dm_control.suite import ALL_TASKS

        all_tasks = ["-".join(task) for task in ALL_TASKS]
        if env_id in all_tasks:
            return "dmc"
    except ModuleNotFoundError:
        pass

    return None
