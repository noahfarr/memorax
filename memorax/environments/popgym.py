import gymnasium.spaces as gym_spaces
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces

from memorax.utils.typing import Array, Key


@struct.dataclass
class PopGymState:
    step: int = 0


class PopGymWrapper:

    def __init__(self, environment):
        self._environment = environment
        self.num_envs = environment.num_envs

        observation_space = environment.single_observation_space
        match observation_space:
            case gym_spaces.Box():
                self.observation_shape = observation_space.shape
                self.observation_dtype = jnp.dtype(observation_space.dtype)
                self.observation_low = np.asarray(observation_space.low)
                self.observation_high = np.asarray(observation_space.high)
            case gym_spaces.Discrete():
                self.observation_shape = ()
                self.observation_dtype = jnp.int32
                self.observation_low = np.asarray(0, dtype=np.int32)
                self.observation_high = np.asarray(int(observation_space.n) - 1, dtype=np.int32)
            case gym_spaces.MultiDiscrete():
                self.observation_shape = observation_space.shape
                self.observation_dtype = jnp.int32
                self.observation_low = np.zeros(observation_space.shape, dtype=np.int32)
                self.observation_high = np.asarray(observation_space.nvec, dtype=np.int32) - 1
            case _:
                raise NotImplementedError(
                    f"Unsupported popgym observation space: {type(observation_space).__name__}"
                )

        self.num_actions = environment.single_action_space.n

    @property
    def default_params(self) -> None:
        return None

    def reset(self, key: Key, params=None) -> tuple[Array, PopGymState]:

        def _reset(key):
            observation, _ = self._environment.reset()
            return jnp.array(observation, dtype=self.observation_dtype)

        observation = jax.pure_callback(
            _reset,
            jax.ShapeDtypeStruct(self.observation_shape, self.observation_dtype),
            key,
            vmap_method="broadcast_all",
        )

        state = PopGymState(step=0)
        return observation, state

    def step(
        self,
        key: Key,
        state: PopGymState,
        action: Array,
        params=None,
    ) -> tuple[Array, PopGymState, Array, Array, dict]:

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

        new_state = PopGymState(step=state.step + 1)
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


def make(env_id, num_envs=1, **kwargs) -> tuple:
    import gymnasium
    import popgym  # noqa: F401  # registers popgym envs with gymnasium

    environment = gymnasium.make_vec(env_id, num_envs=num_envs, **kwargs)
    return PopGymWrapper(environment), None
