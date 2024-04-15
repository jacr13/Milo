import random

import gymnasium as gym
import metaworld

from milo.env.utils import gym_vector_env_creator


def make(env_id: str, num_envs: int = 1, vectorization_mode: str = "async", **kwargs) -> gym.vector.VectorEnv:
    render_mode = kwargs.get("render_mode", "rgb_array")
    vector_kwargs = kwargs.get("vector_kwargs", {})

    ml1 = metaworld.ML1(env_id)
    env_fns = [lambda: ml1.train_classes[env_id](render_mode=render_mode) for _ in range(num_envs)]

    envs = gym_vector_env_creator(env_fns, vectorization_mode, **vector_kwargs)

    # Randomly sample tasks, assumes num_env <= len(train_tasks)
    assert num_envs <= len(ml1.train_tasks), "num_envs must be <= len(train_tasks)"

    tasks = random.sample(ml1.train_tasks, num_envs)
    envs.call_ids("set_task", [{"args": [task]} for task in tasks])

    return envs
