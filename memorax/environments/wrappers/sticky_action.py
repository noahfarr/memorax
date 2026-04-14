from typing import Union

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment

from memorax.utils.typing import Array, Key
from gymnax.wrappers.purerl import GymnaxWrapper


@struct.dataclass
class StickyActionWrapperState:
    previous_action: Array
    env_state: environment.EnvState


class StickyActionWrapper(GymnaxWrapper):
    def __init__(self, env, sticky_action_probability: float = 0.1):
        super().__init__(env)
        self.sticky_action_probability = sticky_action_probability

    def reset(
        self, key: Key, params: environment.EnvParams | None = None
    ) -> tuple[Array, StickyActionWrapperState]:
        observation, env_state = self._env.reset(key, params)
        state = StickyActionWrapperState(
            previous_action=jnp.int32(0), env_state=env_state
        )
        return observation, state

    def step(
        self,
        key: Key,
        state: StickyActionWrapperState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, StickyActionWrapperState, float, bool, dict]:
        key, sticky_key = jax.random.split(key)
        sticky = jax.random.uniform(sticky_key) < self.sticky_action_probability
        executed_action = jnp.where(sticky, state.previous_action, action)
        observation, env_state, reward, done, info = self._env.step(
            key, state.env_state, executed_action, params
        )
        state = StickyActionWrapperState(
            previous_action=executed_action, env_state=env_state
        )
        return observation, state, reward, done, info
