import warnings

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces

from memorax.utils.typing import Array, Key


@struct.dataclass
class PufferLibEnvState:
    step: int = 0


class PufferLibWrapper:

    def __init__(self, env, batch_shape: tuple[int, ...] = (1,)):
        self.env = env
        self.batch_shape = tuple(batch_shape)

        if len(self.batch_shape) > 1:
            warnings.warn(
                f"PufferLibWrapper batch_shape={self.batch_shape} treats leading "
                "axes as seeds, but all envs share a single underlying vec env "
                "and its RNG state, so sub-batches are not independently seeded. "
                "Seed each sub-env explicitly at make-time if you need "
                "independent seeds.",
                stacklevel=2,
            )

        obs_space = env.single_observation_space
        self.obs_shape = obs_space.shape
        self.obs_dtype = jnp.dtype(obs_space.dtype)

        self.num_actions = env.single_action_space.n

    @property
    def default_params(self) -> None:
        return None

    def reset(self, key: Key, params=None) -> tuple[Array, PufferLibEnvState]:

        def _reset(key):
            obs, _ = self.env.reset()
            obs = np.reshape(obs, self.batch_shape + self.obs_shape)
            return jnp.array(obs, dtype=self.obs_dtype)

        obs = jax.pure_callback(
            _reset,
            jax.ShapeDtypeStruct(self.obs_shape, self.obs_dtype),
            key,
            vmap_method="broadcast_all",
        )

        state = PufferLibEnvState(step=0)
        return obs, state

    def step(
        self,
        key: Key,
        state: PufferLibEnvState,
        action: Array,
        params=None,
    ) -> tuple[Array, PufferLibEnvState, Array, Array, dict]:

        def _step(action):
            action = np.reshape(action, (-1,))
            action = np.asarray(action, dtype=np.int32)
            obs, rewards, dones, truncs, infos = self.env.step(action)
            obs = np.reshape(obs, self.batch_shape + self.obs_shape)
            rewards = np.reshape(rewards, self.batch_shape)
            dones = np.reshape(dones | truncs, self.batch_shape)
            return (
                jnp.array(obs, dtype=self.obs_dtype),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(dones, dtype=jnp.bool_),
            )

        obs, rewards, dones = jax.pure_callback(
            _step,
            (
                jax.ShapeDtypeStruct(self.obs_shape, self.obs_dtype),
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.bool_),
            ),
            action,
            vmap_method="broadcast_all",
        )

        new_state = PufferLibEnvState(step=state.step + 1)
        return obs, new_state, rewards, dones, {}

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=self.obs_shape,
            dtype=self.obs_dtype,
        )

    def action_space(self, params=None) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)


def make(env_id, env_creator, batch_shape: tuple[int, ...] = (1,), **kwargs) -> tuple:
    import pufferlib.vector

    num_envs = int(np.prod(batch_shape))
    environment = pufferlib.vector.make(env_creator, num_envs=num_envs, **kwargs)
    return PufferLibWrapper(environment, batch_shape=batch_shape), None
