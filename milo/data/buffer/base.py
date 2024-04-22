import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.reset()

    def reset(self):
        self.buffer = []

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove the first element
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
