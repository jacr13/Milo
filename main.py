from milo.data.collector import Collector
from milo.env import make_env

envs_list = ["Humanoid-v5", "button-press-topdown-v2", "walker-walk"]


for env_name in envs_list:
    env = make_env(env_name, num_envs=3, vectorization_mode="async", env_spec_kwargs={"render_mode": "rgb_array"})
    obs, info = env.reset(seed=13)

    collector = Collector(None, env)

    collector.reset()
    collector.collect(n_step=1000, render=True)

    buffer = collector.buffer

    print(buffer.batchify())

    batch = buffer.sample(10)
    batch.to_torch()

    input("Press Enter to continue...")
