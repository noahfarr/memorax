import warnings

import gymnasium.spaces as gym_spaces
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces

from memorax.utils.typing import Array, Key


@struct.dataclass
class GymnasiumState:
    step: int = 0


class GymnasiumWrapper:

    def __init__(self, environment, batch_shape: tuple[int, ...] = (1,)):
        self.environment = environment
        self.batch_shape = tuple(batch_shape)

        if len(self.batch_shape) > 1:
            warnings.warn(
                f"GymnasiumWrapper batch_shape={self.batch_shape} treats leading "
                "axes as seeds, but all envs share a single underlying vec env "
                "and its RNG state, so sub-batches are not independently seeded. "
                "Seed each sub-env explicitly at make-time if you need "
                "independent seeds.",
                stacklevel=2,
            )

        observation_space = environment.single_observation_space
        self.observation_shape = observation_space.shape
        self.observation_dtype = jax.dtypes.canonicalize_dtype(observation_space.dtype)

        action_space = environment.single_action_space
        self.action_shape = action_space.shape
        self.action_dtype = action_space.dtype

    @property
    def default_params(self) -> None:
        return None

    def reset(self, key: Key, params=None) -> tuple[Array, GymnasiumState]:

        def _reset(key):
            observation, _ = self.environment.reset()
            observation = np.reshape(
                observation, self.batch_shape + self.observation_shape
            )
            return jnp.array(observation, dtype=self.observation_dtype)

        observation = jax.pure_callback(
            _reset,
            jax.ShapeDtypeStruct(self.observation_shape, self.observation_dtype),
            key,
            vmap_method="broadcast_all",
        )

        state = GymnasiumState(step=0)
        return observation, state

    def step(
        self,
        key: Key,
        state: GymnasiumState,
        action: Array,
        params=None,
    ) -> tuple[Array, GymnasiumState, Array, Array, dict]:

        def _step(action):
            action = np.reshape(action, (-1,) + self.action_shape)
            action = np.asarray(action, dtype=self.action_dtype)
            observation, rewards, terminations, truncations, infos = (
                self.environment.step(action)
            )
            observation = np.reshape(
                observation, self.batch_shape + self.observation_shape
            )
            rewards = np.reshape(rewards, self.batch_shape)
            dones = np.reshape(terminations | truncations, self.batch_shape)
            return (
                jnp.array(observation, dtype=self.observation_dtype),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(dones, dtype=jnp.bool_),
            )

        observation, rewards, dones = jax.pure_callback(
            _step,
            (
                jax.ShapeDtypeStruct(self.observation_shape, self.observation_dtype),
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.bool_),
            ),
            action,
            vmap_method="broadcast_all",
        )

        new_state = GymnasiumState(step=state.step + 1)
        return observation, new_state, rewards, dones, {}

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=self.observation_shape,
            dtype=self.observation_dtype,
        )

    def action_space(self, params=None):
        action_space = self.environment.single_action_space
        match action_space:
            case gym_spaces.Discrete(n=n):
                return spaces.Discrete(int(n))
            case gym_spaces.Box(low=low, high=high, shape=shape, dtype=dtype):
                return spaces.Box(np.asarray(low), np.asarray(high), shape, dtype)
        raise NotImplementedError(
            f"Unsupported gymnasium action space: {type(action_space).__name__}"
        )


def make(env_id, batch_shape: tuple[int, ...] = (1,), **kwargs) -> tuple:
    import gymnasium

    num_envs = int(np.prod(batch_shape))
    environment = gymnasium.make_vec(env_id, num_envs=num_envs, **kwargs)
    return GymnasiumWrapper(environment, batch_shape=batch_shape), None
