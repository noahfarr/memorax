from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from memorax.utils import Timestep, Transition, periodic_incremental_update, utils
from memorax.utils.axes import (
    add_feature_axis,
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils.typing import (
    Array,
    Buffer,
    BufferState,
    Carry,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class DQNConfig:
    num_envs: int
    gamma: float
    tau: float
    target_update_frequency: int
    train_frequency: int
    burn_in_length: int = 0


@struct.dataclass(frozen=True)
class DQNState:
    step: int
    update_step: int
    timestep: Timestep
    carry: tuple
    env_state: EnvState
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: BufferState


@dataclass
class DQN:
    cfg: DQNConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    optimizer: optax.GradientTransformation
    buffer: Buffer
    epsilon_schedule: optax.Schedule

    def __post_init__(self):
        assert (
            self.cfg.train_frequency >= self.cfg.num_envs
        ), f"train_frequency ({self.cfg.train_frequency}) must be >= num_envs ({self.cfg.num_envs})"
        assert (
            self.cfg.train_frequency % self.cfg.num_envs == 0
        ), f"train_frequency ({self.cfg.train_frequency}) must be divisible by num_envs ({self.cfg.num_envs})"

    def _greedy_action(self, key: Key, state: DQNState) -> tuple[DQNState, Array, dict]:
        torso_key = key
        (carry, (q_values, _)), intermediates = self.q_network.apply(
            state.params,
            *state.timestep.to_sequence(),
            initial_carry=state.carry,
            rngs={"torso": torso_key},
            mutable=["intermediates"],
        )
        action = jnp.argmax(q_values, axis=-1)
        action = remove_time_axis(action)
        state = state.replace(carry=carry)
        return state, action, intermediates

    def _random_action(self, key: Key, state: DQNState) -> tuple[DQNState, Array, dict]:
        action_key = jax.random.split(key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return state, action, {}

    def _epsilon_greedy_action(
        self, key: Key, state: DQNState
    ) -> tuple[DQNState, Array, dict]:
        random_key, greedy_key, sample_key = jax.random.split(key, 3)

        state, random_action, _ = self._random_action(random_key, state)
        state, greedy_action, intermediates = self._greedy_action(greedy_key, state)

        epsilon = self.epsilon_schedule(state.step)
        action = jnp.where(
            jax.random.uniform(sample_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        return state, action, intermediates

    def _step(
        self, state: DQNState, key: Key, *, policy: Callable
    ) -> tuple[DQNState, Transition]:
        action_key, step_key = jax.random.split(key)

        initial_carry = state.carry

        state, action, intermediates = policy(action_key, state)
        num_envs, *_ = state.timestep.obs.shape
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        intermediates = jax.tree.map(
            lambda x: jnp.mean(jnp.stack(x)),
            intermediates.get("intermediates", {}),
            is_leaf=lambda x: isinstance(x, tuple),
        )
        lox.log({"intermediates": intermediates, "info": info})

        first = Timestep(
            obs=state.timestep.obs,
            action=jnp.where(
                state.timestep.done,
                jnp.zeros_like(state.timestep.action),
                state.timestep.action,
            ),
            reward=jnp.where(
                state.timestep.done,
                jnp.zeros_like(state.timestep.reward),
                state.timestep.reward,
            ),
            done=state.timestep.done,
        ).to_sequence()
        second = Timestep(
            obs=next_obs,
            action=action,
            reward=reward,
            done=done,
        ).to_sequence()
        transition = Transition(
            first=first,
            second=second,
            carry=jax.tree.map(add_time_axis, initial_carry),
        )

        buffer_state = self.buffer.add(state.buffer_state, transition)

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(
                obs=next_obs,
                action=action,
                reward=jnp.asarray(reward, dtype=jnp.float32),
                done=done,
            ),
            env_state=env_state,
            buffer_state=buffer_state,
        )
        return state, transition

    def _update(self, key: Key, state: DQNState) -> DQNState:
        batch_key, torso_key, next_torso_key = jax.random.split(key, 3)
        batch = self.buffer.sample(state.buffer_state, batch_key)

        experience = batch.experience
        experience = jax.tree.map(lambda x: jnp.expand_dims(x, 1), experience)

        initial_carry = None
        initial_target_carry = None
        if experience.carry is not None:
            initial_carry = jax.tree.map(lambda x: x[:, 0], experience.carry)
            initial_target_carry = jax.tree.map(lambda x: x[:, 0], experience.carry)

        initial_carry = utils.burn_in(self.q_network, state.params, experience.first, initial_carry, self.cfg.burn_in_length)
        initial_target_carry = utils.burn_in(self.q_network, state.target_params, experience.second, initial_target_carry, self.cfg.burn_in_length)
        experience = jax.tree.map(lambda x: x[:, self.cfg.burn_in_length:], experience)

        _, (next_target_q_values, _) = self.q_network.apply(
            state.target_params,
            *experience.second,
            initial_carry=initial_target_carry,
            rngs={"torso": next_torso_key},
        )
        next_target_q_value = jnp.max(next_target_q_values, axis=-1)

        td_target = (
            experience.second.reward
            + self.cfg.gamma * (1 - experience.second.done) * next_target_q_value
        )

        def loss_fn(params: PyTree):
            carry, (q_values, aux) = self.q_network.apply(
                params,
                *experience.first,
                initial_carry=initial_carry,
                rngs={"torso": torso_key},
            )
            action = add_feature_axis(experience.second.action)
            q_value = jnp.take_along_axis(q_values, action, axis=-1)
            q_value = remove_feature_axis(q_value)
            td_error = q_value - td_target
            loss = self.q_network.head.loss(
                q_value, aux, td_target, transitions=experience
            ).mean()
            return loss, (q_value, td_error, carry)

        (loss, (q_value, td_error, carry)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        lox.log({"q_network/gradient_norm": optax.global_norm(grads)})
        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        params = optax.apply_updates(state.params, updates)
        target_params = periodic_incremental_update(
            params,
            state.target_params,
            state.step,
            self.cfg.target_update_frequency,
            self.cfg.tau,
        )

        lox.log(
            {
                "q_network/loss": loss,
                "q_network/q_value": q_value.mean(),
                "q_network/td_error": td_error.mean(),
                "training/epsilon": self.epsilon_schedule(state.step),
                "training/step": state.step,
                "training/update_step": state.update_step,
            }
        )

        state = state.replace(
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
        )

        return state

    def _update_step(self, state: DQNState, key: Key) -> tuple[DQNState, None]:
        step_key, update_key = jax.random.split(key)

        step_keys = jax.random.split(
            step_key, self.cfg.train_frequency // self.cfg.num_envs
        )
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            state,
            step_keys,
        )

        state = self._update(update_key, state)

        return state.replace(update_step=state.update_step + 1), None

    def init(self, key: Key) -> DQNState:
        env_key, q_key, torso_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        carry = self.q_network.initialize_carry((self.cfg.num_envs, None))

        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()
        params = target_params = self.q_network.init(
            {"params": q_key, "torso": torso_key},
            *timestep,
            initial_carry=carry,
        )
        optimizer_state = self.optimizer.init(params)

        timestep = timestep.from_sequence()
        transition = Transition(
            first=timestep,
            second=timestep,
            carry=carry,
        )
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

        return DQNState(
            step=0,
            update_step=0,
            timestep=timestep,
            carry=carry,
            env_state=env_state,
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            buffer_state=buffer_state,
        )

    def warmup(self, key: Key, state: DQNState, num_steps: int) -> DQNState:
        step_keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            state,
            step_keys,
        )
        return state

    def train(
        self,
        key: Key,
        state: DQNState,
        num_steps: int,
    ) -> DQNState:
        num_outer_steps = num_steps // self.cfg.train_frequency
        keys = jax.random.split(key, num_outer_steps)
        state, _ = jax.lax.scan(
            self._update_step,
            state,
            keys,
        )

        return state

    def evaluate(self, key: Key, state: DQNState, num_steps: int) -> DQNState:
        reset_key, eval_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        carry = self.q_network.initialize_carry((self.cfg.num_envs, None))

        state = state.replace(timestep=timestep, carry=carry, env_state=env_state)

        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._greedy_action),
            state,
            step_keys,
        )

        return state
