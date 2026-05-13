from typing import Callable, Union

from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper

from memorax.utils.typing import Array, Key


class TransformRewardWrapper(GymnaxWrapper):
    def __init__(self, env, func: Callable[[float], float]):
        super().__init__(env)
        self.func = func

    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, environment.EnvState, float, bool, dict]:
        observation, env_state, reward, done, info = self._env.step(
            key, state, action, params
        )
        return observation, env_state, self.func(reward), done, info
