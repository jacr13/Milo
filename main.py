import time

import numpy as np

from milo.data.collector import Collector
from milo.env import make_env
from milo.env.dmc import DMC2Gym


env = DMC2Gym(domain="walker", task="walk", task_kwargs={"random": 13})
obs_13 = env.action_space.sample()
print(obs_13)

env = DMC2Gym(domain="walker", task="walk", task_kwargs={"random": 22})
obs_13 = env.action_space.sample()
print(obs_13)

env = DMC2Gym(domain="walker", task="walk", task_kwargs={"random": 13})
obs_13 = env.action_space.sample()
print(obs_13)