from milo.data.collector import Collector
from milo.env import make_env

# gymnasium
env = make_env("Humanoid-v5", num_envs=3, vectorization_mode="async", env_spec_kwargs={"render_mode": "rgb_array"})
obs, info = env.reset(seed=13)
imgs = env.render()
print(obs.shape, len(imgs), imgs[0].shape, imgs[1].shape, imgs[2].shape, type(info))

# metaworld
env = make_env("button-press-topdown-v2", num_envs=2, vectorization_mode="sync")
obs, info = env.reset(seed=13)
imgs = env.render()
print(obs.shape, len(imgs), imgs[0].shape, imgs[1].shape)

# dmc
env = make_env("walker-walk", num_envs=3, vectorization_mode="sync")
obs, info = env.reset(seed=13)
for i in range(10):
    obs, _, _, _, _ = env.step(env.action_space.sample())
    imgs = env.render()
    print(i, obs.shape, len(imgs), imgs[0].shape, imgs[1].shape, imgs[2].shape)


print()
print()
print()
print()
env = make_env("Humanoid-v5", num_envs=3, vectorization_mode="async", env_spec_kwargs={"render_mode": "rgb_array"})
collector = Collector(env.action_space, env, buffer=None, exploration_noise=False)
collector.reset(gym_reset_kwargs={"seed": 123})
print(collector._pre_obs)

collector.reset(seed=13)
print(collector._pre_obs)

collector.reset(seed=123)
print(collector._pre_obs)
print(collector.env_num)
collector.collect(n_step=10)
