from milo.data.collector import Collector
from milo.env import make_env

envs_list = ["Humanoid-v5", "button-press-topdown-v2", "walker-walk"]


# for env_name in envs_list:

env_name = "Humanoid-v5"
train_env = make_env(env_name, num_envs=3, vectorization_mode="async", env_spec_kwargs={"render_mode": None})
test_env = make_env(env_name, num_envs=2, vectorization_mode="async", env_spec_kwargs={"render_mode": None})


# logger = WandbLogger(
#     experiment_name="test_milo_exp",
#     project="test_milo",
#     group="test_milo",
#     config={"env": env_name, "policy": "random"},
#     log_dir="logs",
# )

train_collector = Collector(None, train_env)

train_collector.reset()
train_collector.collect(n_step=1000, reset_before_collect=True)

buffer = train_collector.buffer
batch = buffer.to_batch()

print(batch)

# batch = train_collector.buffer.sample(10)

# for i in range(10):
#     print("iter", i)
#     for batch in train_collector.buffer.batches(batch_size=500):
#         print(batch)

# # input("Press Enter to continue...")
