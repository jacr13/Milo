from typing import Any

from gymnasium.error import AlreadyPendingCallError
from gymnasium.vector.async_vector_env import AsyncState
from gymnasium.vector.async_vector_env import AsyncVectorEnv as OriginalAsyncVectorEnv
from gymnasium.vector.sync_vector_env import SyncVectorEnv as OriginalSyncVectorEnv


def gym_vector_env_creator(env_fns: list, vectorization_mode: str, **vector_kwargs):
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


class SyncVectorEnv(OriginalSyncVectorEnv):
    def call_ids(self, name: str, args_ids: list) -> tuple[Any, ...]:
        """Calls a sub-environment method with name and specific kwargs.

        Returns:
            Tuple of results
        """
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
        """Call a method from each parallel environment with idependent args
        Args:
            name (str): Name of the method or property to call.

        Returns:
            List of the results of the individual calls to the method or property for each environment.
        """
        self.call_async_ids(name, *args, **kwargs)
        return self.call_wait()

    def call_async_ids(self, name: str, args_ids: list):
        """Calls the method with name asynchronously and
        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_async` while waiting for a pending call to complete
        """
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
        self._state = AsyncState.WAITING_CALL
