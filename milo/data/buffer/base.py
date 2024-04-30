import random

from milo.data.batch import Batch
from milo.data.transition import Transition


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: list = []
        self.reset()

    def reset(self) -> None:
        self.buffer = []

    def push(self, transition: Transition) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove the first element
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Batch:
        batch = random.sample(self.buffer, batch_size)
        return Batch(batch)

    def batchify(self) -> Batch:
        return Batch(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, buffer length={self.__len__()})"
