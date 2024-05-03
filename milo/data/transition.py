import numpy as np


class Transition:
    def __init__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        terminated: np.ndarray | None = None,
        truncated: np.ndarray | None = None,
        info: dict | None = None,
        pixels: tuple | np.ndarray | None = None,
    ) -> None:
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.pixels = pixels

    def __repr__(self) -> str:
        return (
            f"Transition(\n"
            f"  obs={self.obs},\n"
            f"  action={self.action},\n"
            f"  reward={self.reward},\n"
            f"  next_obs={self.next_obs},\n"
            f"  done={self.done},\n"
            f"  terminated={self.terminated},\n"
            f"  truncated={self.truncated},\n"
            ")"
        )
