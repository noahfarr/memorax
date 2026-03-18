import numpy as np
from gymnax.environments import spaces

from memorax.environments.wrappers import GymnaxWrapper
from memorax.utils.typing import Array, Key


class NavixGymnaxWrapper(GymnaxWrapper):
    def reset(self, key: Key, params=None) -> tuple[Array, Array]:
        timestep = self._env.reset(key)
        return timestep.observation, timestep

    def step(self, key: Key, state, action: Array, params=None) -> tuple[Array, Array, Array, Array, dict]:
        timestep = self._env.step(state, action)
        return timestep.observation, timestep, timestep.reward, timestep.is_done(), {}

    def observation_space(self, params) -> spaces.Box:
        return spaces.Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=(np.prod(self._env.observation_space.shape),),
            dtype=self._env.observation_space.dtype,
        )

    def action_space(self, params) -> spaces.Discrete:
        return spaces.Discrete(
            num_categories=self._env.action_space.maximum.item() + 1,
        )


def make(env_id: str, **kwargs) -> tuple:
    import navix as nx

    env = nx.make(env_id, **kwargs)
    env = NavixGymnaxWrapper(env)
    return env, None
