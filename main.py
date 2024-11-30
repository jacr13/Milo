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
import numpy as np
import torch

gamma: float = 0.99
lam: float = 0.95

rewards = batch.reward  # Shape: (T, N)
values = np.zeros_like(rewards)
dones = batch.done  # Shape: (T, N)

print(values.shape)

if isinstance(rewards, torch.Tensor):
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[1], device=rewards.device)
else:  # NumPy
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(rewards.shape[1])

print(rewards.shape, values.shape, dones.shape)
print(returns.shape, advantages.shape, last_gae.shape)

# Iterate backwards to compute returns and advantages
for t in reversed(range(rewards.shape[0])):
    mask = 1.0 - dones[t]  # Handle episode ends
    next_value = values[t + 1] if t + 1 < rewards.shape[0] else 0
    delta = rewards[t] + gamma * next_value * mask - values[t]
    last_gae = delta + gamma * lam * mask * last_gae
    advantages[t] = last_gae
    returns[t] = advantages[t] + values[t]

# Store the computed values in the batch
batch.returns = returns
batch.advantages = advantages

print(returns)
print(advantages)

last_gae = np.zeros(buffer.buffer[0].reward.shape[1])

values

T = len(buffer.buffer)
for t in reversed(range(T)):
    transition = buffer.buffer[t]
    mask = 1.0 - transition.done
    next_value = values if t + 1 < T else 0
    delta = transition.reward + gamma * next_value * mask - values
    last_gae = delta + gamma * lam * mask * last_gae
    advantages[t] = last_gae
    returns[t] = advantages[t] + values[t]

# batch = train_collector.buffer.sample(10)

# for i in range(10):
#     print("iter", i)
#     for batch in train_collector.buffer.batches(batch_size=500):
#         print(batch)

# # input("Press Enter to continue...")
