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

    def __init__(self, environment):
        self._environment = environment
        self.num_envs = environment.num_envs

        observation_space = environment.single_observation_space
        self.observation_shape = observation_space.shape
        self.observation_dtype = jnp.dtype(observation_space.dtype)

        self.num_actions = environment.single_action_space.n

    @property
    def default_params(self) -> None:
        return None

    def reset(self, key: Key, params=None) -> tuple[Array, GymnasiumState]:

        def _reset(key):
            observation, _ = self._environment.reset()
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
            action = np.asarray(action, dtype=np.int32)
            observation, rewards, terminations, truncations, infos = (
                self._environment.step(action)
            )

            return (
                jnp.array(observation, dtype=self.observation_dtype),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(terminations | truncations, dtype=jnp.bool_),
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

    def action_space(self, params=None) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)


def make(env_id, num_envs=1, **kwargs) -> tuple:
    import gymnasium

    environment = gymnasium.make_vec(env_id, num_envs=num_envs, **kwargs)
    return GymnasiumWrapper(environment), None
