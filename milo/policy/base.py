import os
import os.path as osp
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Literal, cast

import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete

from milo.data.batch import Batch, BatchAction, BatchObs
from milo.data.buffer.base import ReplayBuffer
from milo.utils.timer import Timer


class BasePolicy(ABC, nn.Module):
    _action_type: Literal["discrete", "continuous"] | None = None
    updating: bool = False

    def __init__(
        self,
        *,
        action_space: gym.Space,
        observation_space: gym.Space | None = None,
        action_scaling: bool = False,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: None = None,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self._action_type = self.action_type
        self.action_scaling = action_scaling
        self.action_bound_method = action_bound_method
        self.lr_scheduler = lr_scheduler

    @property
    def action_type(self) -> Literal["discrete", "continuous"]:
        if self._action_type is not None:
            return self._action_type
        # If action_type is not yet known, infer it from action_space
        if isinstance(self.action_space, Discrete | MultiDiscrete | MultiBinary):
            action_type = "discrete"
        elif isinstance(self.action_space, Box):
            action_type = "continuous"
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}.")
        return cast(Literal["discrete", "continuous"], action_type)

    def compute_action(self):
        pass

    @abstractmethod
    def forward(self, batch: BatchObs, **kwargs: Any) -> BatchAction:
        """Forward logic for policy (should be implemented by subclasses)."""

    @abstractmethod
    def learn(self, batch: Batch, *args: Any, **kwargs: Any) -> None:
        """Learn logic for policy (should be implemented by subclasses)."""

    def process_fn(self, batch: Batch, buffer: ReplayBuffer) -> Batch:
        """Process batch data before learning (should be implemented by subclasses)."""
        return batch

    def post_process_fn(self, batch: Batch, buffer: ReplayBuffer):
        """Post-process batch data after learning (should be implemented by subclasses)."""

    @contextmanager
    def updating_context(self):
        timer = Timer().start()
        try:
            self.updating = True
            yield timer
        finally:
            timer.stop()
            self.updating = False

    def update(self, sample_size: int | None, buffer: ReplayBuffer | None, **kwargs: Any) -> None:
        if buffer is None:
            raise ValueError("Buffer is None, can not update policy.")

        with self.updating_context() as timer:
            # Sample batch from buffer
            batch = buffer.sample(sample_size)

            # Process batch data before learning
            batch = self.process_fn(batch, buffer)

            # Learn
            self.learn(batch, **kwargs)

            # Post-process batch data after learning
            self.post_process_fn(batch, buffer)

            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        print(f"Update time: {timer.final_time()}")

    def save(self, path: str, filename: str = "policy.pt") -> None:
        os.makedirs(path, exist_ok=True)
        objects_to_save = {}

        for attr_name, attr_value in self.__dict__.items():
            if hasattr(attr_value, "state_dict"):
                state_dict = attr_value.state_dict()
                if len(state_dict) > 0:
                    objects_to_save[attr_name] = state_dict

        if len(objects_to_save) > 0:
            torch.save(objects_to_save, osp.join(path, filename))

    def load(self, path: str | None = None, state_dict: dict | None = None, filename: str = "policy.pt") -> None:
        objects_loaded = state_dict or None

        if objects_loaded is None and path is not None:
            objects_loaded = torch.load(osp.join(path, filename))
        else:
            raise ValueError("Must provide either path or state_dict.")

        assert isinstance(objects_loaded, dict), "Invalid state dict."
        for attr_name, attr_value in objects_loaded.items():
            if attr_name in self.__dict__:
                self.__dict__[attr_name].load_state_dict(attr_value)
