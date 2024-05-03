from milo.data.buffer.base import ReplayBuffer
from milo.data.collector import Collector
from milo.env import make_env

envs_list = ["Humanoid-v5", "button-press-topdown-v2", "walker-walk"]


for env_name in envs_list:
    env = make_env(env_name, num_envs=3, vectorization_mode="async", env_spec_kwargs={"render_mode": "rgb_array"})

    buffer = ReplayBuffer(1000000)
    collector = Collector(None, env, buffer=buffer)

    collector.reset()
    collector.collect(n_step=1000, render=False)

    print(buffer.to_batch(only=["obs"]))

    batch = buffer.sample(10)
    batch.to_torch()
    print(batch.obs.shape)

    input("Press Enter to continue...")
