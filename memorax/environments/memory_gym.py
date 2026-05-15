import warnings

import gymnasium.spaces as gym_spaces
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces

from memorax.utils.typing import Array, Key


@struct.dataclass
class MemoryGymState:
    step: int = 0


class MemoryGymWrapper:

    def __init__(self, environment, batch_shape: tuple[int, ...] = (1,)):
        self.environment = environment
        self.batch_shape = tuple(batch_shape)

        if len(self.batch_shape) > 1:
            warnings.warn(
                f"MemoryGymWrapper batch_shape={self.batch_shape} treats leading "
                "axes as seeds, but all envs share a single underlying vec env "
                "and its RNG state, so sub-batches are not independently seeded. "
                "Seed each sub-env explicitly at make-time if you need "
                "independent seeds.",
                stacklevel=2,
            )

        observation_space = environment.single_observation_space
        self.observation_shape = observation_space.shape
        self.observation_dtype = jnp.dtype(observation_space.dtype)
        self.observation_low = np.asarray(observation_space.low)
        self.observation_high = np.asarray(observation_space.high)

        action_space = environment.single_action_space
        self.action_shape = action_space.shape
        self.action_dtype = action_space.dtype
        match action_space:
            case gym_spaces.Discrete(n=n):
                self.num_actions = int(n)
                self.action_nvec = None
            case gym_spaces.MultiDiscrete(nvec=nvec):
                self.action_nvec = np.asarray(nvec)
                self.num_actions = int(np.prod(self.action_nvec))
            case _:
                raise NotImplementedError(
                    f"Unsupported memory_gym action space: {type(action_space).__name__}"
                )

    @property
    def default_params(self) -> None:
        return None

    def reset(self, key: Key, params=None) -> tuple[Array, MemoryGymState]:

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

        state = MemoryGymState(step=0)
        return observation, state

    def step(
        self,
        key: Key,
        state: MemoryGymState,
        action: Array,
        params=None,
    ) -> tuple[Array, MemoryGymState, Array, Array, dict]:

        def _step(action):
            action = np.reshape(action, (-1,))
            if self.action_nvec is not None:
                action = np.stack(
                    np.unravel_index(action, self.action_nvec), axis=-1
                )
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

        new_state = MemoryGymState(step=state.step + 1)
        return observation, new_state, rewards, dones, {}

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(
            low=self.observation_low,
            high=self.observation_high,
            shape=self.observation_shape,
            dtype=self.observation_dtype,
        )

    def action_space(self, params=None) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)


def make(env_id, batch_shape: tuple[int, ...] = (1,), **kwargs) -> tuple:
    import gymnasium
    import memory_gym  # noqa: F401  # registers memory-gym envs with gymnasium

    num_envs = int(np.prod(batch_shape))
    environment = gymnasium.make_vec(env_id, num_envs=num_envs, **kwargs)
    return MemoryGymWrapper(environment, batch_shape=batch_shape), None
