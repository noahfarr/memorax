from typing import Optional, Tuple, Union

import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment

from memorax.utils.typing import Array, Key
from gymnax.wrappers.purerl import GymnaxWrapper


@struct.dataclass
class NormalizeObservationWrapperState:
    mean: jnp.ndarray
    M2: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeObservationWrapper(GymnaxWrapper):
    def __init__(self, env, eps: float = 1e-8):
        super().__init__(env)
        self.eps = eps

    def _welford_update(self, mean, M2, count, obs):
        count = count + 1
        delta = obs - mean
        mean = mean + delta / count
        delta2 = obs - mean
        M2 = M2 + delta * delta2
        return mean, M2, count

    def reset(
        self, key: Key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[Array, NormalizeObservationWrapperState]:
        obs, env_state = self._env.reset(key, params)
        mean = jnp.zeros_like(obs)
        M2 = jnp.ones_like(obs)
        count = 1.0
        mean, M2, count = self._welford_update(mean, M2, count, obs)
        state = NormalizeObservationWrapperState(
            mean=mean,
            M2=M2,
            count=count,
            env_state=env_state,
        )
        return (obs - mean) / jnp.sqrt(M2 / count + self.eps), state

    def step(
        self,
        key: Key,
        state: NormalizeObservationWrapperState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, NormalizeObservationWrapperState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        mean, M2, count = self._welford_update(
            state.mean, state.M2, state.count, obs
        )
        state = NormalizeObservationWrapperState(
            mean=mean,
            M2=M2,
            count=count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.M2 / state.count + self.eps),
            state,
            reward,
            done,
            info,
        )
