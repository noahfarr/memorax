from typing import Any

import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from memorax.environments.wrappers import GymnaxWrapper
from memorax.utils.typing import Array, Key


@struct.dataclass(frozen=True)
class EnvParams:
    env_params: Any
    max_steps_in_episode: int


class XLandMiniGridWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key: Key, params) -> tuple[Array, Array]:
        timestep = self._env.reset(params.env_params, key)

        return timestep.observation, timestep

    def step(self, key: Key, state, action: Array, params) -> tuple[Array, Array, Array, Array, dict]:

        timestep = self._env.step(params.env_params, state, action)
        done = timestep.step_type == 2
        return timestep.observation, timestep, timestep.reward, done, {}

    def observation_space(self, params) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_shape(params.env_params),),
        )

    def action_space(self, params) -> spaces.Discrete:
        return spaces.Discrete(self._env.num_actions(params.env_params))


def make(env_id: str, benchmark_name=None, ruleset_key=None, **kwargs) -> tuple:
    import xminigrid
    from xminigrid.wrappers import GymAutoResetWrapper

    env, env_params = xminigrid.make(env_id, **kwargs)

    if benchmark_name is not None:
        benchmark = xminigrid.load_benchmark(name=benchmark_name)
        ruleset = benchmark.sample_ruleset(ruleset_key)
        env_params = env_params.replace(ruleset=ruleset)

    max_steps_in_episode = env_params.max_steps

    env = GymAutoResetWrapper(env)
    env = XLandMiniGridWrapper(env)
    env_params = EnvParams(
        env_params=env_params, max_steps_in_episode=max_steps_in_episode
    )
    return env, env_params
