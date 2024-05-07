import warnings
from typing import Any, Literal

import numpy as np
import torch

from milo.data.transition import Transition


class Batch:
    obs: np.ndarray | torch.Tensor | None = None
    action: np.ndarray | torch.Tensor | None = None
    reward: np.ndarray | torch.Tensor | None = None
    next_obs: np.ndarray | torch.Tensor | None = None
    done: np.ndarray | torch.Tensor | None = None
    terminated: np.ndarray | torch.Tensor | None = None
    truncated: np.ndarray | torch.Tensor | None = None
    info: list | None = None
    pixels: np.ndarray | torch.Tensor | None = None

    def __init__(
        self,
        data: list | dict,
        batch_type: Literal["torch", "numpy"] | None = None,
        device: torch.device | None = None,
        exclude: list | None = None,
        only: list | None = None,
    ) -> None:
        assert exclude is None or only is None, "Cannot specify both exclude and only keys."
        exclude = exclude or []
        only = only or []

        self._device = device or torch.device("cpu")
        self._keys = self._setup_keys(exclude, only)
        self._data = data
        self._batch_type = batch_type or self._get_batch_type(data)
        self._setup(set_attr=True)

    def _get_batch_type(self, data: list | dict) -> Literal["torch", "numpy"]:
        data_example = None
        if isinstance(data, list) and isinstance(data[0], Transition):
            data_example = data[0].obs
        if isinstance(data, dict):
            key = next(iter(data.keys()))
            data_example = data[key]

        if data_example is None:
            raise ValueError("Data should be a list of transitions or a dictionary of arrays.")

        if isinstance(data_example, torch.Tensor):
            return "torch"
        if isinstance(data_example, np.ndarray):
            return "numpy"
        raise ValueError("Data should be a list of transitions or a dictionary of arrays.")

    def _setup_keys(self, exclude: list, only: list) -> list:
        if only:
            return only

        # Get the keys of the Batch object
        keys = list(self.__annotations__.keys())
        if exclude:
            return list(set(keys) - set(exclude))
        return keys

    def get_keys(self) -> list:
        return self._keys

    def set_key(self, key: str, value: np.ndarray | None = None) -> None:
        setattr(self, key, value)

    def _setup(self, set_attr: bool = False) -> dict:
        batch_dict = {}
        if isinstance(self._data, list) and isinstance(self._data[0], Transition):
            # Check if all transitions have the same attributes
            for transition in self._data[1:]:
                assert set(self._keys).issubset(
                    set(transition.__dict__.keys()),
                ), f"Transitions should have at least the following keys: {self._keys}, but got {set(transition.__dict__.keys())}"

            # Stack the attributes of the transitions in numpy arrays
            batch_dict = {key: [getattr(transition, key) for transition in self._data] for key in self._keys}
        elif isinstance(self._data, dict):
            assert set(self._keys).issubset(
                self._data.keys(),
            ), f"Data should have at least the following keys: {self._keys}, but got {self._data.keys()}"
            batch_dict = {key: self._data[key] for key in self._keys}
        else:
            raise ValueError(f"Unsupported data type: {type(self._data)}")

        if self._batch_type == "numpy":
            batch_dict = {key: np.stack(value) for key, value in batch_dict.items()}
        elif self._batch_type == "torch":
            batch_dict = {
                key: torch.stack(value) if isinstance(value, list) else value for key, value in batch_dict.items()
            }

        # Set the attributes of the Batch object to the numpy arrays (optional)
        if set_attr:
            for key, value in batch_dict.items():
                setattr(self, key, value)

        return batch_dict

    def to_numpy(self, exclude: list | None = None, only: list | None = None) -> None:
        assert only is None or exclude is None, "Cannot specify both exclude and only keys."
        exclude = exclude or []
        keys = only or list(self.__dict__.keys())

        for key in keys:
            if isinstance(self.__dict__[key], np.ndarray) or key in exclude:
                continue

            if torch and isinstance(self.__dict__[key], torch.Tensor):
                self.__dict__[key] = self.__dict__[key].cpu().numpy()
            else:
                self.__dict__[key] = np.array(self.__dict__[key])

        self._batch_type = "numpy"

    def to_torch(
        self, device: torch.device | None = None, exclude: list | None = None, only: list | None = None,
    ) -> None:
        assert only is None or exclude is None, "Cannot specify both exclude and only keys."
        exclude = exclude or ["info"]
        keys = only or list(self.__dict__.keys())

        device = device or torch.device("cpu")

        for key in keys:
            # Skip the keys in exclude
            if key in exclude or key.startswith("_"):
                continue

            if isinstance(self.__dict__[key], torch.Tensor):
                self.__dict__[key] = self.__dict__[key].to(device)

            if isinstance(self.__dict__[key], np.ndarray):
                # Check if the dtype is not an object (impossible to convert to torch)
                # Needed to avoid converting arrays of Nones for exemple
                if self.__dict__[key].dtype == np.object_:
                    continue

                self.__dict__[key] = torch.from_numpy(self.__dict__[key]).to(device)
            else:
                try:
                    self.__dict__[key] = torch.tensor(self.__dict__[key], device=device)
                except TypeError as exc:
                    raise ValueError(f"Unknown type: {type(self.__dict__[key])}") from exc

        self._batch_type = "torch"

    def to(self, device: torch.device) -> None:
        if self._batch_type == "numpy":
            warnings.warn("Batch is in numpy format, it will be converted to torch format automatically.")

        self.to_torch(device=device)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        attributes = ",\n".join(
            [f"  {key} = {self.__dict__[key].shape if self.__dict__[key] is not None else None}" for key in self._keys],
        )
        return f"Batch(\n{attributes}\n)"


class BatchObs(Batch):
    """Batch of observations."""
    def __init__(self, obs: np.ndarray | torch.Tensor, **kwargs: Any) -> None:
        super().__init__({"obs": obs}, only=["obs"], **kwargs)

    def __repr__(self) -> str:
        assert self._keys == ["obs"], "Only the 'obs' key is supported."
        attributes = ",\n".join(
            [f"  {key} = {self.__dict__[key].shape if self.__dict__[key] is not None else None}" for key in self._keys],
        )
        return f"BatchObs(\n{attributes}\n)"


class BatchAction(Batch):
    """Batch of actions."""
    def __init__(self, action: np.ndarray | torch.Tensor, **kwargs: Any) -> None:
        super().__init__({"action": action}, only=["action"], **kwargs)

    def __repr__(self) -> str:
        assert self._keys == ["action"], "Only the 'action' key is supported."
        attributes = ",\n".join(
            [f"  {key} = {self.__dict__[key].shape if self.__dict__[key] is not None else None}" for key in self._keys],
        )
        return f"BatchAction(\n{attributes}\n)"
