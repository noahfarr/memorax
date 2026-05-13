from typing import Union

import lox
from gymnax.environments import environment

from memorax.utils.typing import Array, Key
from gymnax.wrappers.purerl import GymnaxWrapper


class ScaleRewardWrapper(GymnaxWrapper):
    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)
        lox.log({"scale_reward/scale": self.scale})
        return obs, env_state, self.scale * reward, done, info
