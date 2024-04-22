import random

from milo.data.transition import Transition

class ReplayBuffer:
    def __init__(self, capacity: int)->None:
        self.capacity = capacity
        self.buffer: list = []
        self.reset()

    def reset(self) -> None:
        self.buffer = []

    def push(self, transition: Transition) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove the first element
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
