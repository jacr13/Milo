from milo.env import make_env

# gymnasium
env = make_env("Humanoid-v5", num_envs=3, vectorization_mode="async")
obs, info = env.reset(seed=13)
print(obs)

# metaworld
env = make_env("button-press-topdown-v2", num_envs=2, vectorization_mode="sync")
obs, info = env.reset(seed=13)
print(obs)

# dmc
env = make_env("walker-walk", num_envs=3, vectorization_mode="sync")
obs, info = env.reset(seed=13)
print(obs)