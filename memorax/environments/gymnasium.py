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

    def __init__(self, environment, num_seeds: int = 1, num_envs: int = 1):
        self.environment = environment
        self.num_seeds = num_seeds
        self.num_envs = num_envs

        observation_space = environment.single_observation_space
        self.observation_shape = observation_space.shape
        self.observation_dtype = jax.dtypes.canonicalize_dtype(observation_space.dtype)

        action_space = environment.single_action_space
        self.action_shape = action_space.shape
        self.action_dtype = action_space.dtype

        if num_seeds > 1:
            self.batch_shape = (num_seeds, num_envs)
        else:
            self.batch_shape = (num_envs,)

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


def make(env_id, num_seeds: int = 1, num_envs: int = 1, **kwargs) -> tuple:
    import gymnasium

    environment = gymnasium.make_vec(
        env_id, num_envs=num_seeds * num_envs, **kwargs
    )
    return (
        GymnasiumWrapper(environment, num_seeds=num_seeds, num_envs=num_envs),
        None,
    )
