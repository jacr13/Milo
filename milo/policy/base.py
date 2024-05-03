from typing import Literal

import torch.nn as nn


class BasePolicy(nn.Module):
    def __init__(
        self,
        *,
        action_space,
        observation_space: None = None,
        action_scaling: bool = False,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: None = None,
    ) -> None:
        self._action_type: Literal["discrete", "continuous"] = "discrete"

    @property
    def action_type(self) -> Literal["discrete", "continuous"]:
        return self._action_type

    def compute_action(self):
        pass

    def forward(self):
        pass

    def learn(self):
        pass

    def update(self):
        pass

    @staticmethod
    def compute_episodic_return():
        return

    def save(self, path: str):
        pass

    def load(self, path: str | None = None, state_dict: dict | None = None):
        pass
