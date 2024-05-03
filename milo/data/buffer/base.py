import numpy as np

from milo.data.batch import Batch
from milo.data.transition import Transition


class ReplayBuffer:
    _seed: int | None = None
    _random: np.random.Generator = np.random.default_rng()

    def __init__(self, capacity: int | None = None, seed: int | None = None) -> None:
        self._seed = seed
        self.capacity = capacity
        self.buffer: list = []
        self.reset()

    def seed(self, seed: int | None) -> None:
        self._seed = seed
        self._random = np.random.default_rng(seed)

    def reset(self) -> None:
        self.buffer = []

    def push(self, transition: Transition) -> None:
        if self.capacity is not None and len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove the first element if the buffer is full
        self.buffer.append(transition)

    def sample(
        self, batch_size: int, replace: bool = False, exclude: list | None = None, only: list | None = None,
    ) -> Batch:
        batch = self._random.choice(self.buffer, batch_size, replace=replace).tolist()
        return Batch(batch, exclude=exclude, only=only)

    def to_batch(self, exclude: list | None = None, only: list | None = None) -> Batch:
        return Batch(self.buffer, exclude=exclude, only=only)

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, buffer length={self.__len__()})"
