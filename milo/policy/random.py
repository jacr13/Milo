from typing import Any

import gymnasium as gym
import numpy as np

from milo.data.batch import Batch, BatchAction, BatchObs
from milo.policy.base import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, action_space: gym.Space):
        super().__init__(action_space=action_space)

    def forward(self, batch: BatchObs, **kwargs: Any) -> BatchAction:
        obs = batch.obs
        assert obs is not None, "Batch does not contain observation."

        act = [self.action_space.sample() for _ in range(obs.shape[0])]
        return BatchAction(np.array(act))

    def learn(self, batch: Batch, *args: Any, **kwargs: Any) -> None:
        pass
