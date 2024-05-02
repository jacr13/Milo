import numpy as np

from milo.data.batch import Batch
from milo.data.transition import Transition


class ReplayBuffer:
    _seed: int | None = None
    _random: np.random.Generator = np.random.default_rng()

    def __init__(self, capacity: int, seed: int | None = None) -> None:
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
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove the first element
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Batch:
        batch = self._random.choice(self.buffer, batch_size, replace=False).tolist()
        return Batch(batch)

    def batchify(self) -> Batch:
        return Batch(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, buffer length={self.__len__()})"
