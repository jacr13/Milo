class Transition:
    def __init__(self, obs, action, reward, next_obs, done, terminated=None, truncated=None, info=None, pixels=None):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.pixels = pixels

    def __repr__(self):
        return (
            f"Transition(obs={self.obs}, action={self.action}, "
            f"reward={self.reward}, next_obs={self.next_obs}, "
            f"done={self.done}, terminated={self.terminated}, "
            f"truncated={self.truncated})"
        )
