from typing import Union

import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment

from memorax.utils.typing import Array, Key
from gymnax.wrappers.purerl import GymnaxWrapper


@struct.dataclass
class TimeAwareObservationWrapperState:
    step: int
    env_state: environment.EnvState


class TimeAwareObservationWrapper(GymnaxWrapper):
    def __init__(self, env, normalize: bool = False, max_steps: int | None = None):
        super().__init__(env)
        if normalize and max_steps is None:
            raise ValueError("max_steps must be provided when normalize=True")
        self.normalize = normalize
        self.max_steps = max_steps

    def _augment(self, observation: Array, step: int) -> Array:
        time = jnp.asarray(step, dtype=observation.dtype)
        if self.normalize:
            time = time / jnp.asarray(self.max_steps, dtype=observation.dtype) - 0.5
        return jnp.concatenate([observation, time[None]], axis=-1)

    def reset(
        self, key: Key, params: environment.EnvParams | None = None
    ) -> tuple[Array, TimeAwareObservationWrapperState]:
        observation, env_state = self._env.reset(key, params)
        state = TimeAwareObservationWrapperState(step=0, env_state=env_state)
        return self._augment(observation, 0), state

    def step(
        self,
        key: Key,
        state: TimeAwareObservationWrapperState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, TimeAwareObservationWrapperState, float, bool, dict]:
        observation, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_step = jnp.where(done, 0, state.step + 1)
        new_state = TimeAwareObservationWrapperState(
            step=new_step, env_state=env_state
        )
        return self._augment(observation, new_step), new_state, reward, done, info
