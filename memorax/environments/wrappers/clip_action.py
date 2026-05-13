from typing import Union

import jax.numpy as jnp
import lox
from gymnax.environments import environment

from memorax.utils.typing import Array, Key
from gymnax.wrappers.purerl import GymnaxWrapper


class ClipActionWrapper(GymnaxWrapper):
    def __init__(self, env, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, environment.EnvState, float, bool, dict]:
        clip_rate = jnp.mean(
            jnp.logical_or(action < self.low, action > self.high).astype(jnp.float32)
        )
        lox.log({"clip_action/clip_rate": clip_rate})
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)
