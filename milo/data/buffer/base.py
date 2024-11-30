from collections.abc import Iterator

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
        self._idx_to_sample = None
        self._idx_current = 0

    def push(self, transition: Transition) -> None:
        if self.capacity is not None and len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove the first element if the buffer is full
        self.buffer.append(transition)

    def batches(
        self,
        batch_size: int,
        flatten: bool = True,
        exclude: list | None = None,
        only: list | None = None,
    ) -> Iterator[Batch]:
        batch_size = batch_size or len(self.buffer)

        buffer = self.buffer
        if flatten:
            buffer = [*transision.flatten() for transition in self.buffer]

        idx_to_sample = list(range(len(buffer)))
        self._random.shuffle(idx_to_sample)

        # Iterate over the shuffled indices in chunks of `batch_size`
        for start in range(0, len(idx_to_sample), batch_size):
            batch_indices = idx_to_sample[start : start + batch_size]
            batch = [buffer[idx] for idx in batch_indices]
            yield Batch(batch)

    def sample(
        self,
        batch_size: int | None,
        replace: bool = False,
        exclude: list | None = None,
        only: list | None = None,
    ) -> Batch:
        batch_size = batch_size or len(self.buffer)

        # TODO: check if we need to return indices
        # indices = self._random.choice(len(self.buffer), batch_size, replace=replace)
        # batch = np.array(self.buffer)[indices].tolist()
        # otherwise return the batch simply from choice
        batch = self._random.choice(self.buffer, batch_size, replace=replace).tolist()
        return Batch(batch, exclude=exclude, only=only)

    def to_batch(self, exclude: list | None = None, only: list | None = None) -> Batch:
        return Batch(self.buffer, exclude=exclude, only=only)

    def compute_returns_and_advantages(self) -> (None, None):
        returns, advantages = [], []
        return returns, advantages

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, buffer_length={self.__len__()})"
