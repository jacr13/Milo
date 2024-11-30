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
        returns: np.ndarray | None = None,
        advantages: np.ndarray | None = None,
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
        self.returns = returns
        self.advantages = advantages
        self.info = info
        self.pixels = pixels

    def unpack(self):
        """Unpack this Transition into a list of smaller Transitions based on the first dimension."""
        num_transitions = self.obs.shape[0]
        unpacked_transitions = []

        for i in range(num_transitions):
            new_transition = Transition(
                obs=self.obs[i],
                action=self.action[i],
                reward=self.reward[i],
                next_obs=self.next_obs[i],
                done=self.done[i],
                terminated=self.terminated[i] if self.terminated is not None else None,
                truncated=self.truncated[i] if self.truncated is not None else None,
                returns=self.returns[i] if self.returns is not None else None,
                advantages=self.advantages[i] if self.advantages is not None else None,
                info={key: value[i] for key, value in self.info.items()},
                pixels=self.pixels[i] if self.pixels is not None else None,
            )
            unpacked_transitions.append(new_transition)

        return unpacked_transitions

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
            f"  returns={self.returns},\n"
            f"  advantages={self.advantages},\n"
            ")"
        )
