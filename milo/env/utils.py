from typing import Any

from gymnasium.error import AlreadyPendingCallError
from gymnasium.vector.async_vector_env import AsyncState
from gymnasium.vector.async_vector_env import AsyncVectorEnv as OriginalAsyncVectorEnv
from gymnasium.vector.sync_vector_env import SyncVectorEnv as OriginalSyncVectorEnv


class SyncVectorEnv(OriginalSyncVectorEnv):
    def call_ids(self, name: str, args_ids: list) -> tuple[Any, ...]:
        """Synchronously calls a method by name on all environments and returns the results."""
        assert len(args_ids) == len(self.envs), "number of args_ids must match number of envs"
        results = []
        for i, env in enumerate(self.envs):
            function = env.get_wrapper_attr(name)

            if callable(function):
                assert (
                    "args" in args_ids[i] or "kwargs" in args_ids[i]
                ), f"args or kwargs must be provided for i {i} env"
                args = args_ids[i].get("args", [])
                kwargs = args_ids[i].get("kwargs", {})
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)


class AsyncVectorEnv(OriginalAsyncVectorEnv):
    def call_ids(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Asynchronously calls a method by name on all environments and waits for the result."""
        self.call_async_ids(name, *args, **kwargs)
        return self.call_wait()

    def call_async_ids(self, name: str, args_ids: list) -> None:
        """A method to asynchronously call multiple functions with their respective arguments through pipes."""
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `call_async` while waiting for a pending call to `{self._state.value}` to complete.",
                str(self._state.value),
            )

        assert len(self.parent_pipes) == len(args_ids), "Number of parent_pipes and args_ids must match"

        for i, pipe in enumerate(self.parent_pipes):
            assert "args" in args_ids[i] or "kwargs" in args_ids[i], f"args or kwargs must be provided for i {i} env"
            args = args_ids[i].get("args", [])
            kwargs = args_ids[i].get("kwargs", {})
            pipe.send(("_call", (name, args, kwargs)))
        self._state: AsyncState = AsyncState.WAITING_CALL


def gym_vector_env_creator(
    env_fns: list,
    vectorization_mode: str,
    vector_kwargs: dict | None = None,
) -> SyncVectorEnv | AsyncVectorEnv:
    """Create a vectorized environment from a list of environment functions based on the specified vectorization mode."""
    vector_kwargs = vector_kwargs or {}

    if vectorization_mode == "async":
        return AsyncVectorEnv(
            env_fns=env_fns,
            **vector_kwargs,
        )
    elif vectorization_mode == "sync":
        return SyncVectorEnv(
            env_fns=env_fns,
            **vector_kwargs,
        )
    else:
        raise ValueError(f"Invalid vectorization mode {vectorization_mode}.")
