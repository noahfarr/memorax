from typing import Any

from flax import struct

from memorax.environments.wrappers import GymnaxWrapper
from memorax.utils.typing import Array, Key


@struct.dataclass(frozen=True)
class EnvParams:
    env_params: Any
    max_steps_in_episode: int


class PopGymArcadeWrapper(GymnaxWrapper):
    def reset(self, key: Key, params) -> tuple[Array, Array]:
        return self._env.reset(key, params.env_params)

    def step(self, key: Key, state, action: Array, params) -> tuple[Array, Array, Array, Array, dict]:
        return self._env.step(key, state, action, params.env_params)


def make(env_id: str, **kwargs) -> tuple:
    import popgym_arcade

    env, env_params = popgym_arcade.make(env_id, **kwargs)
    env = PopGymArcadeWrapper(env)

    env_params = EnvParams(
        env_params=env_params, max_steps_in_episode=env._env.max_steps_in_episode
    )
    return env, env_params
