import numpy as np
import torch


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

    def __init__(self, batch: list) -> None:
        self._batch = batch

        self.batchify(set_attr=True)

    def batchify(self, set_attr: bool = False) -> dict:
        keys = list(self._batch[0].__dict__.keys())

        # Check if all transitions have the same attributes
        for transition in self._batch[1:]:
            assert set(transition.__dict__.keys()) == set(
                keys,
            ), "All transitions in the batch must have the same attributes."

        # Stack the attributes of the transitions in numpy arrays
        batch_dict = {key: np.stack([getattr(transition, key) for transition in self._batch]) for key in keys}

        # Set the attributes of the Batch object to the numpy arrays (optional)
        if set_attr:
            for key, value in batch_dict.items():
                setattr(self, key, value)

        return batch_dict

    def to_torch(self, device: str = "cpu", exclude_keys: list | None = None) -> None:
        exclude_keys = exclude_keys or ["info"]

        for key in self.__dict__:
            if isinstance(self.__dict__[key], np.ndarray) and key not in exclude_keys:
                self.__dict__[key] = torch.from_numpy(self.__dict__[key]).to(device)

    def to_numpy(self) -> None:
        for key in self.__dict__:
            if isinstance(self.__dict__[key], torch.Tensor):
                self.__dict__[key] = self.__dict__[key].cpu().numpy()

    def __len__(self) -> int:
        return len(self._batch)

    def __repr__(self) -> str:

        return (
            "Batch(\n"
            f"\tobs = {self.obs if self.obs is None else self.obs.shape},\n"
            f"\taction = {self.action if self.action is None else self.action.shape},\n"
            f"\treward = {self.reward if self.reward is None else self.reward.shape},\n"
            f"\tnext_obs = {self.next_obs if self.next_obs is None else self.next_obs.shape},\n"
            f"\tdone = {self.done if self.done is None else self.done.shape},\n"
            f"\tterminated = {self.terminated if self.terminated is None else self.terminated.shape},\n"
            f"\ttruncated = {self.truncated if self.truncated is None else self.truncated.shape},\n"
            f"\tinfo = {self.info if self.info is None else self.info.shape},\n"
            f"\tpixels = {self.pixels if self.pixels is None else self.pixels.shape},\n"
            ")"
        )
