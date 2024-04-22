# inspired from https://github.com/imgeorgiev/dmc2gymnasium/blob/main/dmc2gymnasium/DMCGym.py

import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from dm_control import suite
from dm_env import specs
from gymnasium import Env, Wrapper
from gymnasium.core import RenderFrame
from gymnasium.spaces import Box

from milo.env.utils import gym_vector_env_creator


def make(
    env_id: str,
    num_envs: int = 1,
    vectorization_mode: str = "async",
    env_spec_kwargs: dict[str, Any] | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    wrappers: Sequence[Callable[[Env], Wrapper]] | None = None,
) -> gym.vector.VectorEnv:
    """Creates a dmc vectorized environment based on the given environment ID, number of environments, vectorization mode, environment specific arguments, vector arguments, and wrappers."""
    env_spec_kwargs = env_spec_kwargs or {}
    vector_kwargs = vector_kwargs or {}
    wrappers = wrappers or []

    domain, task = env_id.split("-")

    env_fns = [lambda: DMC2Gym(domain=domain, task=task, **env_spec_kwargs) for _ in range(num_envs)]

    return gym_vector_env_creator(env_fns, vectorization_mode, **vector_kwargs)


def _spec_to_box(spec: OrderedDict | list, dtype: type = np.float32) -> Box:
    """Converts a specification of observation or action space to a gym Box space."""

    def extract_min_max(s: specs.Array) -> tuple:
        """Takes a specs array and return the minimum and maximum values based on the type."""
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if isinstance(s, specs.BoundedArray):
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        elif isinstance(s, specs.Array):
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        else:
            logging.error("Unrecognized type")
            return None, None

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return Box(low, high, dtype=dtype)


def _flatten_obs(obs: dict[str, np.ndarray | float | int], dtype: type = np.float32) -> np.ndarray:
    """A function that flattens a dictionary of numpy arrays, floats, or integers into a single numpy array."""
    obs_pieces = [v.ravel() if isinstance(v, np.ndarray) else np.array([v]) for v in obs.values()]
    return np.concatenate(obs_pieces, axis=0).astype(dtype)


class DMC2Gym(Env):
    """Converts a dmc environment to a gym compatible environment."""

    def __init__(
        self,
        domain: str,
        task: str,
        task_kwargs: dict | None = None,
        environment_kwargs: dict | None = None,
        render_mode: str = "rgb_array",
        rendering: str = "osmesa",
        render_height: int = 222,
        render_width: int = 480,
        render_camera_id: int = 0,
    ):
        environment_kwargs = environment_kwargs or {}
        task_kwargs = task_kwargs or {}

        # TODO: this seems to be present before importing dm_control suite to avoid warning
        # for details see https://github.com/deepmind/dm_control
        assert rendering in ["glfw", "egl", "osmesa"]
        os.environ["MUJOCO_GL"] = rendering

        self._env = suite.load(
            domain,
            task,
            task_kwargs,
            environment_kwargs,
        )

        # placeholder to allow built in gymnasium rendering
        self.render_mode = render_mode
        self.render_height = render_height
        self.render_width = render_width
        self.render_camera_id = render_camera_id

        self._observation_space = _spec_to_box(self._env.observation_spec().values())
        self._action_space = _spec_to_box([self._env.action_spec()])

        # set seed if provided with task_kwargs
        if "random" in task_kwargs:
            seed = task_kwargs.get("random")
            self._observation_space.seed(seed)
            self._action_space.seed(seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def observation_space(self) -> Box:  # type: ignore
        return self._observation_space

    @property
    def action_space(self) -> Box:  # type: ignore
        return self._action_space

    @property
    def reward_range(self) -> tuple:
        return 0, 1

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Executes a step in the environment."""
        if action.dtype.kind == "f":
            action = action.astype(np.float32)
        assert self._action_space.contains(action)
        timestep = self._env.step(action)
        observation = _flatten_obs(timestep.observation)
        reward = timestep.reward
        termination = False  # we never reach a goal
        truncation = timestep.last()
        info = {"discount": timestep.discount}
        return observation, reward, termination, truncation, info

    def reset(
        self,
        seed: int | np.random.RandomState | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the environment to its initial state depending on the seed and options."""
        if options:
            logging.warning(f"Currently doing nothing with options={options}")

        if seed is not None:
            if isinstance(seed, int):
                # seed also in the action and observation space
                self._observation_space.seed(seed)
                self._action_space.seed(seed)

            if not isinstance(seed, np.random.RandomState):
                seed = np.random.RandomState(seed)
            self._env.task._random = seed

        timestep = self._env.reset()
        observation = _flatten_obs(timestep.observation)
        info: dict[str, Any] = {}
        return observation, info

    def render(
        self,
        height: int | None = None,
        width: int | None = None,
        camera_id: int | None = None,
    ) -> RenderFrame | list[RenderFrame] | None:
        """Renders the current state of the environment."""
        height = height or self.render_height
        width = width or self.render_width
        camera_id = camera_id or self.render_camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
