from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment

from memorax.utils.typing import Array, EnvParams, Key


@struct.dataclass
class RecordEpisodeStatisticsState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class RecordEpisodeStatistics:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(
        self, key: Key, params: EnvParams | None = None
    ) -> tuple[Array, RecordEpisodeStatisticsState]:
        obs, env_state = self._env.reset(key, params)
        state = RecordEpisodeStatisticsState(env_state, 0, 0, 0, 0)
        return obs, state

    def step(
        self,
        key: Key,
        state: RecordEpisodeStatisticsState,
        action: int | float,
        params: EnvParams | None = None,
    ) -> tuple[Array, RecordEpisodeStatisticsState, Array, bool, dict[str, Any]]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = RecordEpisodeStatisticsState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info
