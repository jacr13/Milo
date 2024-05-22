from milo.data.collector import Collector
from milo.env import make_env
from milo.policy.random import RandomPolicy
from milo.trainer.base import Trainer
from milo.utils.logger.wandb import WandbLogger

envs_list = ["Humanoid-v5", "button-press-topdown-v2", "walker-walk"]


# for env_name in envs_list:

env_name = "Humanoid-v5"
train_env = make_env(env_name, num_envs=3, vectorization_mode="async", env_spec_kwargs={"render_mode": "rgb_array"})
test_env = make_env(env_name, num_envs=2, vectorization_mode="async", env_spec_kwargs={"render_mode": "rgb_array"})


logger = WandbLogger(
    experiment_name="test_milo_exp",
    project="test_milo",
    group="test_milo",
    config={"env": env_name, "policy": "random"},
    log_dir="logs",
)

policy = RandomPolicy(train_env.action_space)

train_collector = Collector(policy, train_env)
test_collector = Collector(policy, test_env)

trainer = Trainer(
    policy,
    train_collector,
    test_collector,
    max_epoch=1000,
    batch_size=64,
    step_per_epoch=1000,
    repeat_per_collect=1,
    step_per_collect=2000,
    episode_per_test=11,
    eval_frequency=10,
    save_frequency=100,
    logger=logger,
)

training_stats = trainer.run()

trainer.save()

trainer.load()


# input("Press Enter to continue...")
