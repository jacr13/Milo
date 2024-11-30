import os
import os.path as osp
from abc import ABC, abstractmethod
from typing import Literal, cast

import gym
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete


class BasePolicy(ABC, nn.Module):
    """
    Abstract base class for RL policies.

    Attributes:
        _action_type (Literal["discrete", "continuous"]): The type of actions (inferred or specified).
        updating (bool): Flag to indicate if the policy is currently being updated.
    """

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
        """
        Initialize the BasePolicy.

        Args:
            action_space (gym.Space): The action space of the policy.
            observation_space (gym.Space, optional): The observation space of the policy.
            action_scaling (bool, optional): Whether to scale actions. Defaults to False.
            action_bound_method (Literal["clip", "tanh"], optional): Method to bound actions. Defaults to "clip".
            lr_scheduler (optional): Learning rate scheduler. Defaults to None.
        """
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_scaling = action_scaling
        self.action_bound_method = action_bound_method
        self.lr_scheduler = lr_scheduler

    @property
    def action_type(self) -> Literal["discrete", "continuous"]:
        """
        Determines the action type based on the action space.

        Returns:
            Literal["discrete", "continuous"]: The type of actions.
        """
        if self._action_type is not None:
            return self._action_type

        # Infer action type from the action space
        if isinstance(self.action_space, (Discrete, MultiDiscrete, MultiBinary)):
            action_type = "discrete"
        elif isinstance(self.action_space, Box):
            action_type = "continuous"
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}.")

        return cast(Literal["discrete", "continuous"], action_type)

    @abstractmethod
    def train(self) -> None:
        """
        Abstract method for training the policy. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def evaluate(self) -> None:
        """
        Abstract method for evaluating the policy. Must be implemented by subclasses.
        """
        pass

    def save(self, path: str, filename: str = "policy.pt", verbose: bool = False) -> None:
        """
        Saves the policy state_dict and other components.

        Args:
            path (str): Directory to save the policy.
            filename (str): Filename for the saved file. Defaults to "policy.pt".
            verbose (bool): Whether to print information. Defaults to False.
        """
        os.makedirs(path, exist_ok=True)
        objects_to_save = {}

        # Save components with state_dict
        for attr_name, attr_value in self.__dict__.items():
            if hasattr(attr_value, "state_dict"):
                state_dict = attr_value.state_dict()
                if state_dict:  # Only save non-empty state dicts
                    objects_to_save[attr_name] = state_dict

        if objects_to_save:
            torch.save(objects_to_save, osp.join(path, filename))
            print(f"Policy saved to {osp.join(path, filename)}")
        else:
            print("No components with state_dict found to save.")

    def load(
        self,
        path: str | None = None,
        state_dict: dict | None = None,
        filename: str = "policy.pt",
        verbose: bool = False,
    ) -> None:
        """
        Loads the policy state_dict and other components.

        Args:
            path (str, optional): Directory to load the policy from.
            state_dict (dict, optional): Preloaded state_dict to use. Defaults to None.
            filename (str): Filename for the saved file. Defaults to "policy.pt".
            verbose (bool): Whether to print information. Defaults to False.

        Raises:
            ValueError: If neither `path` nor `state_dict` is provided.
            AssertionError: If the loaded data is not a dictionary.
        """
        objects_loaded = state_dict

        if objects_loaded is None:
            if path is not None:
                objects_loaded = torch.load(osp.join(path, filename))
            else:
                raise ValueError("Must provide either `path` or `state_dict` to load from.")

        assert isinstance(objects_loaded, dict), "Invalid state dict loaded."

        # Load state_dict into corresponding components
        for attr_name, attr_value in objects_loaded.items():
            if attr_name in self.__dict__ and hasattr(self.__dict__[attr_name], "load_state_dict"):
                self.__dict__[attr_name].load_state_dict(attr_value)
                print(f"Loaded {attr_name} from state_dict.")
