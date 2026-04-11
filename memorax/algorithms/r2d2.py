from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flashbax.utils import get_tree_shape_prefix
from flax import core, struct

from memorax.buffers import compute_importance_weights
from memorax.utils import Timestep, Transition, periodic_incremental_update, utils
from memorax.utils.axes import add_feature_axis, remove_feature_axis, remove_time_axis
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
class R2D2Config:
    num_envs: int
    gamma: float
    tau: float
    target_update_frequency: int
    train_frequency: int
    burn_in_length: int = 10
    sequence_length: int = 80
    n_step: int = 5
    priority_exponent: float = 0.9
    importance_sampling_exponent: float = 0.6


@struct.dataclass(frozen=True)
class R2D2State:
    step: int
    update_step: int
    timestep: Timestep
    carry: tuple
    env_state: EnvState
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: BufferState


def compute_n_step_returns(
    rewards: Array,
    dones: Array,
    next_q_values: Array,
    n_step: int,
    gamma: float,
) -> Array:
    batch_size, sequence_length = rewards.shape
    num_targets = sequence_length - n_step + 1

    def compute_target(start_idx: int):
        n_step_return = jnp.zeros(batch_size)
        discount = 1.0
        done = jnp.ones(batch_size)

        for i in range(n_step):
            idx = start_idx + i
            n_step_return = n_step_return + discount * rewards[:, idx] * done
            discount = discount * gamma
            done = done * (1.0 - dones[:, idx])

        bootstrap_idx = start_idx + n_step - 1
        n_step_return = (
            n_step_return + discount * next_q_values[:, bootstrap_idx] * done
        )

        return n_step_return

    targets = jax.vmap(compute_target)(jnp.arange(num_targets))
    targets = targets.T

    return targets


@dataclass
class R2D2:
    cfg: R2D2Config
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    optimizer: optax.GradientTransformation
    buffer: Buffer
    epsilon_schedule: optax.Schedule
    beta_schedule: optax.Schedule

    def __post_init__(self):
        assert self.cfg.train_frequency >= self.cfg.num_envs, (
            f"train_frequency ({self.cfg.train_frequency}) must be >= num_envs ({self.cfg.num_envs})"
        )
        assert self.cfg.train_frequency % self.cfg.num_envs == 0, (
            f"train_frequency ({self.cfg.train_frequency}) must be divisible by num_envs ({self.cfg.num_envs})"
        )
        assert self.cfg.sequence_length > self.cfg.burn_in_length, (
            f"sequence_length ({self.cfg.sequence_length}) must be > burn_in_length ({self.cfg.burn_in_length})"
        )

    def _greedy_action(
        self, key: Key, state: R2D2State
    ) -> tuple[R2D2State, Array, Array, dict]:
        torso_key = key
        obs, done, action, reward = state.timestep.to_sequence()
        (carry, (q_values, _)), intermediates = self.q_network.apply(
            state.params,
            observation=obs,
            done=done,
            action=action,
            reward=reward,
            initial_carry=state.carry,
            rngs={"torso": torso_key},
            mutable=["intermediates"],
        )
        q_values = remove_time_axis(q_values)
        action = jnp.argmax(q_values, axis=-1)
        state = state.replace(carry=carry)
        return state, action, q_values, intermediates

    def _random_action(
        self, key: Key, state: R2D2State
    ) -> tuple[R2D2State, Array, None, dict]:
        action_key = jax.random.split(key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return state, action, None, {}

    def _epsilon_greedy_action(
        self, key: Key, state: R2D2State
    ) -> tuple[R2D2State, Array, Array, dict]:
        random_key, greedy_key, sample_key = jax.random.split(key, 3)

        state, random_action, _, _ = self._random_action(random_key, state)
        state, greedy_action, q_values, intermediates = self._greedy_action(greedy_key, state)

        epsilon = self.epsilon_schedule(state.step)
        action = jnp.where(
            jax.random.uniform(sample_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        return state, action, q_values, intermediates

    def _step(self, state: R2D2State, key: Key, *, policy: Callable) -> tuple[R2D2State, Transition]:
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

        first = Timestep(
            obs=state.timestep.obs,
            action=jnp.where(
                state.timestep.done,
                jnp.zeros_like(state.timestep.action),
                state.timestep.action,
            ),
            reward=jnp.where(
                state.timestep.done,
                0,
                state.timestep.reward,
            ),
            done=state.timestep.done,
        )
        second = Timestep(
            obs=next_obs,
            action=action,
            reward=reward,
            done=done,
        )
        lox.log({"info": info, "intermediates": intermediates})

        transition = Transition(
            first=first,
            second=second,
            carry=initial_carry,
        )

        buffer_transition = jax.tree.map(lambda x: jnp.expand_dims(x, 1), transition)
        buffer_state = self.buffer.add(state.buffer_state, buffer_transition)

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

    def _update(self, key: Key, state: R2D2State):
        sample_key, torso_key, next_torso_key = jax.random.split(key, 3)
        batch = self.buffer.sample(state.buffer_state, sample_key)

        experience = batch.experience

        initial_carry = None
        initial_target_carry = None
        if experience.carry is not None:
            initial_carry = jax.tree.map(lambda x: x[:, 0], experience.carry)
            initial_target_carry = jax.tree.map(lambda x: x[:, 0], experience.carry)

        initial_carry = utils.burn_in(self.q_network, state.params, experience.first, initial_carry, self.cfg.burn_in_length)
        initial_target_carry = utils.burn_in(self.q_network, state.target_params, experience.second, initial_target_carry, self.cfg.burn_in_length)
        experience = jax.tree.map(lambda x: x[:, self.cfg.burn_in_length:], experience)

        next_obs, next_done, next_action, next_reward = experience.second
        _, (next_target_q_values, _) = self.q_network.apply(
            state.target_params,
            observation=next_obs,
            done=next_done,
            action=next_action,
            reward=next_reward,
            initial_carry=initial_target_carry,
            rngs={"torso": next_torso_key},
        )

        next_target_q_value = jnp.max(next_target_q_values, axis=-1)

        _, sequence_length = experience.second.reward.shape
        if self.cfg.n_step > 1 and sequence_length >= self.cfg.n_step:
            n_step_targets = compute_n_step_returns(
                experience.second.reward,
                experience.second.done,
                next_target_q_value,
                self.cfg.n_step,
                self.cfg.gamma,
            )
            _, num_targets = n_step_targets.shape
            experience = jax.tree.map(lambda x: x[:, :num_targets], experience)
            td_target = n_step_targets
        else:
            td_target = (
                experience.second.reward
                + self.cfg.gamma * (1 - experience.second.done) * next_target_q_value
            )

        beta = self.beta_schedule(state.step)
        add_batch_size, max_length_time_axis = get_tree_shape_prefix(
            state.buffer_state.experience, n_axes=2
        )
        buffer_capacity = add_batch_size * max_length_time_axis
        buffer_size = jnp.where(
            state.buffer_state.is_full,
            buffer_capacity,
            state.buffer_state.current_index * add_batch_size,
        )
        buffer_size = jnp.maximum(buffer_size, 1)
        importance_weights = compute_importance_weights(
            batch.probabilities, buffer_size, beta
        )
        importance_weights = importance_weights[:, None]

        first_obs, first_done, first_action, first_reward = experience.first

        def loss_fn(params: PyTree):
            carry, (q_values, aux) = self.q_network.apply(
                params,
                observation=first_obs,
                done=first_done,
                action=first_action,
                reward=first_reward,
                initial_carry=initial_carry,
                rngs={"torso": torso_key},
            )
            action = add_feature_axis(experience.second.action)
            q_value = jnp.take_along_axis(q_values, action, axis=-1)
            q_value = remove_feature_axis(q_value)
            td_error = q_value - td_target

            loss = (
                importance_weights
                * self.q_network.head.loss(
                    q_value, aux, td_target, transitions=experience
                )
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

        mean_td_error = jnp.abs(td_error).mean(axis=1)
        new_priorities = mean_td_error + 1e-6
        buffer_state = self.buffer.set_priorities(
            state.buffer_state, batch.indices, new_priorities
        )

        info = {
            "q_network/loss": loss,
            "q_network/q_value": q_value.mean(),
            "q_network/td_error": mean_td_error.mean(),
            "training/epsilon": self.epsilon_schedule(state.step),
        }

        state = state.replace(
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            buffer_state=buffer_state,
        )

        return state, info

    def _update_step(self, state: R2D2State, key: Key) -> tuple[R2D2State, None]:
        step_key, update_key = jax.random.split(key)

        step_keys = jax.random.split(step_key, self.cfg.train_frequency // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            state,
            step_keys,
        )

        state, info = self._update(update_key, state)

        lox.log({**info, "training/step": state.step, "training/update_step": state.update_step})

        return state.replace(update_step=state.update_step + 1), None

    def init(self, key: Key) -> R2D2State:
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
        ts_obs, ts_done, ts_action, ts_reward = timestep
        params = self.q_network.init(
            {"params": q_key, "torso": torso_key},
            observation=ts_obs,
            done=ts_done,
            action=ts_action,
            reward=ts_reward,
            initial_carry=carry,
        )
        target_params = params
        optimizer_state = self.optimizer.init(params)

        timestep = timestep.from_sequence()
        transition = Transition(
            first=timestep,
            second=timestep,
            carry=carry,
        )
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

        return R2D2State(
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

    def warmup(self, key: Key, state: R2D2State, num_steps: int) -> R2D2State:
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
        state: R2D2State,
        num_steps: int,
    ) -> R2D2State:
        num_outer_steps = num_steps // self.cfg.train_frequency
        keys = jax.random.split(key, num_outer_steps)
        state, _ = jax.lax.scan(
            self._update_step,
            state,
            keys,
        )

        return state

    def evaluate(self, key: Key, state: R2D2State, num_steps: int) -> R2D2State:
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
