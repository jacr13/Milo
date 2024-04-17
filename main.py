import numpy as np

from milo.data.collector import Collector
from milo.env import make_env

# gymnasium
env = make_env("Humanoid-v5", num_envs=3, vectorization_mode="async")
obs, info = env.reset(seed=13)
for i in range(100):
    obs, _, _, _, _ = env.step(env.action_space.sample())
    imgs = env.render()
print(i, obs.shape, len(imgs), imgs[0].shape, imgs[1].shape, imgs[2].shape)

# metaworld
env = make_env("button-press-topdown-v2", num_envs=2, vectorization_mode="sync")
obs, info = env.reset(seed=13)
for i in range(3):
    obs, _, _, _, _ = env.step(env.action_space.sample())
    imgs = env.render()
print(i, obs.shape, len(imgs), imgs[0].shape, imgs[1].shape)

# dmc
env = make_env("walker-walk", num_envs=3, vectorization_mode="sync")
obs, info = env.reset(seed=13)
for i in range(100):
    obs, _, _, _, _ = env.step(env.action_space.sample())
    imgs = env.render()
print(i, obs.shape, len(imgs), imgs[0].shape, imgs[1].shape, imgs[2].shape)


print()
print()
print()
print()
env = make_env("walker-walk", num_envs=3, vectorization_mode="async")
collector = Collector(env.action_space, env, buffer=None, exploration_noise=False)
collector.reset(gym_reset_kwargs={"seed": 123})
print(collector._pre_obs)

collector.reset(seed=13)
print(collector._pre_obs)

collector.reset(seed=123)
print(collector._pre_obs)
print(collector.env_num)

collector.collect(n_step=2000)
print(np.stack(collector.buffer["obs"]).shape, len(collector.buffer["obs"]))
print(collector.collect_time, collector.collect_step, collector.collect_episode)

collector.reset(seed=123)
collector.collect(n_step=1, render=True)
print(np.stack(collector.buffer["obs"]).shape, len(collector.buffer["obs"]))
print(collector.collect_time, collector.collect_step, collector.collect_episode)
