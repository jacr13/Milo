import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class Batch:
    obs: np.ndarray | None = None
    action: np.ndarray | None = None
    reward: np.ndarray | None = None
    next_obs: np.ndarray | None = None
    done: np.ndarray | None = None
    terminated: np.ndarray | None = None
    truncated: np.ndarray | None = None
    info: np.ndarray | None = None
    pixels: np.ndarray | None = None

    def __init__(self, batch: list, exclude: list | None = None, only: list | None = None) -> None:
        assert exclude is None or only is None, "Cannot specify both exclude and only keys."
        exclude = exclude or []
        only = only or []

        self._keys = self._setup_keys(exclude, only)
        self._batch = batch
        self._setup(set_attr=True)

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
        keys = list(self._batch[0].__dict__.keys())

        # Check if all transitions have the same attributes
        for transition in self._batch[1:]:
            assert set(self._keys).issubset(
                set(transition.__dict__.keys()),
            ), f"Transitions should have at least the following keys: {self._keys}"

        # Stack the attributes of the transitions in numpy arrays
        # TODO: this should depend on the type of the transitions received, eg. numpy arrays or torch tensors
        batch_dict = {key: np.stack([getattr(transition, key) for transition in self._batch]) for key in keys}

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

    def to_torch(self, device: str = "cpu", exclude: list | None = None, only: list | None = None) -> None:
        assert only is None or exclude is None, "Cannot specify both exclude and only keys."
        exclude = exclude or ["info"]
        keys = only or list(self.__dict__.keys())

        assert torch is not None, "PyTorch is not installed"

        for key in keys:
            # Skip the keys in exclude
            if key in exclude or key.startswith("_"):
                continue

            if isinstance(self.__dict__[key], torch.Tensor):
                continue

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

    def __len__(self) -> int:
        return len(self._batch)

    def __repr__(self) -> str:
        attributes = ",\n".join(
            [f"  {key} = {self.__dict__[key].shape if self.__dict__[key] is not None else None}" for key in self._keys],
        )
        return f"Batch(\n{attributes}\n)"
