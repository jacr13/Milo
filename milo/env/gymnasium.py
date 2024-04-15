import gymnasium as gym


def make(env_id: str, num_envs: int = 1, vectorization_mode: str = "async", **kwargs) -> gym.vector.VectorEnv:
    env_spec_kwargs = kwargs.get("env_spec_kwargs", {})
    return gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        **env_spec_kwargs,
    )
