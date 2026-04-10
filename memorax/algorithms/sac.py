from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from memorax.utils import Timestep, Transition, periodic_incremental_update
from memorax.utils.axes import remove_time_axis
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
class SACConfig:
    num_envs: int
    gamma: float
    tau: float
    train_frequency: int
    target_update_frequency: int
    target_entropy_scale: float
    gradient_steps: int = 1
    burn_in_length: int = 0


@struct.dataclass(frozen=True)
class SACState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    buffer_state: BufferState
    actor_params: core.FrozenDict[str, Any]
    critic_params: core.FrozenDict[str, Any]
    critic_target_params: core.FrozenDict[str, Any]
    alpha_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    actor_carry: Array
    critic_carry: Array


@dataclass
class SAC:
    cfg: SACConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    alpha_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    alpha_optimizer: optax.GradientTransformation
    buffer: Buffer

    def __post_init__(self):
        assert self.cfg.train_frequency >= self.cfg.num_envs, (
            f"train_frequency ({self.cfg.train_frequency}) must be >= num_envs ({self.cfg.num_envs})"
        )
        assert self.cfg.train_frequency % self.cfg.num_envs == 0, (
            f"train_frequency ({self.cfg.train_frequency}) must be divisible by num_envs ({self.cfg.num_envs})"
        )
        assert self.cfg.gradient_steps >= 1, (
            f"gradient_steps ({self.cfg.gradient_steps}) must be >= 1"
        )

    def _deterministic_action(self, key: Key, state: SACState):
        sample_key = key
        obs, done, action, reward = state.timestep.to_sequence()
        (next_carry, (dist, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=obs,
            done=done,
            action=action,
            reward=reward,
            initial_carry=state.actor_carry,
            temperature=0.0,
            mutable=["intermediates"],
        )
        action = dist.sample(seed=sample_key)
        action = remove_time_axis(action)
        state = state.replace(actor_carry=next_carry)
        return state, action, intermediates

    def _stochastic_action(self, key: Key, state: SACState):
        sample_key = key
        obs, done, action, reward = state.timestep.to_sequence()
        (next_carry, (dist, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=obs,
            done=done,
            action=action,
            reward=reward,
            initial_carry=state.actor_carry,
            mutable=["intermediates"],
        )
        action = dist.sample(seed=sample_key)
        action = remove_time_axis(action)
        state = state.replace(actor_carry=next_carry)
        return state, action, intermediates

    def _random_action(self, key: Key, state: SACState):
        action_keys = jax.random.split(key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_keys)
        return state, action, {}

    def _step(
        self,
        state: SACState,
        key: Key,
        *,
        policy: Callable,
    ):
        initial_carry = state.actor_carry

        action_key, step_key = jax.random.split(key)

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

        broadcast_dims = tuple(
            range(state.timestep.done.ndim, state.timestep.action.ndim)
        )
        prev_action = jnp.where(
            jnp.expand_dims(state.timestep.done, axis=broadcast_dims),
            jnp.zeros_like(state.timestep.action),
            state.timestep.action,
        )
        prev_reward = jnp.where(state.timestep.done, 0, state.timestep.reward)

        first = Timestep(
            obs=state.timestep.obs,
            action=prev_action,
            reward=prev_reward,
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

    def init(self, key: Key):
        env_key, actor_key, actor_torso_key, critic_key, critic_torso_key, alpha_key = (
            jax.random.split(key, 6)
        )
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

        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()

        actor_carry = self.actor_network.initialize_carry((self.cfg.num_envs, None))
        ts_obs, ts_done, ts_action, ts_reward = timestep
        actor_params = self.actor_network.init(
            {"params": actor_key, "torso": actor_torso_key},
            observation=ts_obs,
            done=ts_done,
            action=ts_action,
            reward=ts_reward,
            initial_carry=actor_carry,
        )
        actor_optimizer_state = self.actor_optimizer.init(actor_params)

        critic_carry = self.critic_network.initialize_carry((self.cfg.num_envs, None))
        critic_params = self.critic_network.init(
            {"params": critic_key, "torso": critic_torso_key},
            observation=ts_obs,
            done=ts_done,
            action=ts_action,
            reward=ts_reward,
            initial_carry=critic_carry,
        )
        critic_target_params = critic_params
        critic_optimizer_state = self.critic_optimizer.init(critic_params)

        alpha_params = self.alpha_network.init(alpha_key)
        alpha_optimizer_state = self.alpha_optimizer.init(alpha_params)

        timestep = timestep.from_sequence()
        transition = Transition(
            first=timestep,
            second=timestep,
            carry=actor_carry,
        )
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

        return SACState(
            step=0,
            update_step=0,
            timestep=timestep,
            actor_carry=actor_carry,
            critic_carry=critic_carry,
            env_state=env_state,
            buffer_state=buffer_state,
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            alpha_params=alpha_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            alpha_optimizer_state=alpha_optimizer_state,
        )

    def _update_alpha(
        self,
        key: Key,
        state: SACState,
        experience: PyTree,
        initial_actor_carry: Carry = None,
    ):
        action_dim, *_ = self.env.action_space(self.env_params).shape
        target_entropy = -self.cfg.target_entropy_scale * action_dim

        first_obs, first_done, first_action, first_reward = experience.first
        _, (dist, _) = self.actor_network.apply(
            state.actor_params,
            observation=first_obs,
            done=first_done,
            action=first_action,
            reward=first_reward,
            initial_carry=initial_actor_carry,
        )

        key, sample_key = jax.random.split(key)
        _, log_probs = dist.sample_and_log_prob(seed=sample_key)

        def alpha_loss_fn(alpha_params: PyTree):
            log_alpha = self.alpha_network.apply(alpha_params)
            alpha = jnp.exp(log_alpha)
            alpha_loss = (alpha * (-log_probs - target_entropy)).mean()
            return alpha_loss, (alpha, alpha_loss)

        (_, (alpha, alpha_loss)), grads = jax.value_and_grad(alpha_loss_fn, has_aux=True)(
            state.alpha_params
        )
        lox.log({
            "losses/alpha/loss": alpha_loss,
            "losses/alpha/value": alpha,
            "alpha/gradient_norm": optax.global_norm(grads),
        })
        updates, optimizer_state = self.alpha_optimizer.update(
            grads, state.alpha_optimizer_state, state.alpha_params
        )
        alpha_params = optax.apply_updates(state.alpha_params, updates)

        state = state.replace(
            alpha_params=alpha_params, alpha_optimizer_state=optimizer_state
        )

        return state

    def _update_actor(
        self,
        key: Key,
        state: SACState,
        experience: PyTree,
        initial_actor_carry: Carry = None,
        initial_critic_carry: Carry = None,
    ):
        log_alpha = self.alpha_network.apply(state.alpha_params)
        alpha = jnp.exp(log_alpha)

        first_obs, first_done, first_action, first_reward = experience.first

        def actor_loss_fn(actor_params):
            carry, (dist, _) = self.actor_network.apply(
                actor_params,
                observation=first_obs,
                done=first_done,
                action=first_action,
                reward=first_reward,
                initial_carry=initial_actor_carry,
            )
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            _, (qs, _) = self.critic_network.apply(
                state.critic_params,
                observation=first_obs,
                done=first_done,
                action=actions,
                reward=first_reward,
                initial_carry=initial_critic_carry,
            )
            q = jnp.minimum(*qs)
            actor_loss = (log_probs * alpha - q).mean()
            return actor_loss, (carry, actor_loss, log_probs)

        (_, (carry, actor_loss, log_probs)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            state.actor_params
        )
        lox.log({
            "actor/loss": actor_loss,
            "actor/entropy": -log_probs.mean(),
            "actor/gradient_norm": optax.global_norm(grads),
        })
        updates, actor_optimizer_state = self.actor_optimizer.update(
            grads, state.actor_optimizer_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, updates)

        state = state.replace(
            actor_params=actor_params, actor_optimizer_state=actor_optimizer_state
        )

        return state, carry

    def _update_critic(
        self,
        key: Key,
        state: SACState,
        experience: PyTree,
        initial_actor_carry: Carry = None,
        initial_critic_carry: Carry = None,
        initial_target_critic_carry: Carry = None,
    ):
        second_obs, second_done, second_action, second_reward = experience.second
        _, (dist, _) = self.actor_network.apply(
            state.actor_params,
            observation=second_obs,
            done=second_done,
            action=second_action,
            reward=second_reward,
            initial_carry=initial_actor_carry,
        )
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

        _, (next_qs, _) = self.critic_network.apply(
            state.critic_target_params,
            observation=second_obs,
            done=second_done,
            action=next_actions,
            reward=second_reward,
            initial_carry=initial_target_critic_carry,
        )
        next_q = jnp.minimum(*next_qs)

        log_alpha = self.alpha_network.apply(state.alpha_params)
        alpha = jnp.exp(log_alpha)
        next_value = next_q - alpha * next_log_probs
        target_q = (
            experience.second.reward
            + self.cfg.gamma * (1 - experience.second.done) * next_value
        )

        target_q = jax.lax.stop_gradient(target_q)

        first_obs, first_done, first_action, first_reward = experience.first

        def critic_loss_fn(critic_params):
            _, (qs, _) = self.critic_network.apply(
                critic_params,
                observation=first_obs,
                done=first_done,
                action=second_action,
                reward=first_reward,
                initial_carry=initial_critic_carry,
            )
            q1, q2 = qs
            critic_loss = (
                self.critic_network.head.loss(
                    q1, {}, target_q, transitions=experience
                ).mean()
                + self.critic_network.head.loss(
                    q2, {}, target_q, transitions=experience
                ).mean()
            )

            return critic_loss, (critic_loss, q1, q2)

        (_, (critic_loss, q1, q2)), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            state.critic_params
        )
        lox.log({
            "critic/loss": critic_loss,
            "critic/q1": q1.mean(),
            "critic/q2": q2.mean(),
            "critic/gradient_norm": optax.global_norm(grads),
        })
        updates, critic_optimizer_state = self.critic_optimizer.update(
            grads, state.critic_optimizer_state, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, updates)

        critic_target_params = periodic_incremental_update(
            critic_params,
            state.critic_target_params,
            state.step,
            self.cfg.target_update_frequency,
            self.cfg.tau,
        )

        state = state.replace(
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            critic_optimizer_state=critic_optimizer_state,
        )
        return state

    def _update(self, key: Key, state: SACState):
        batch_key, critic_key, actor_key, alpha_key = jax.random.split(key, 4)
        experience = self.buffer.sample(state.buffer_state, batch_key).experience
        experience = jax.tree.map(lambda x: jnp.expand_dims(x, 1), experience)

        initial_actor_carry = None
        initial_critic_carry = None
        initial_target_critic_carry = None

        if experience.carry is not None:
            initial_actor_carry = jax.tree.map(lambda x: x[:, 0], experience.carry)

        if self.cfg.burn_in_length > 0:
            burn_in = jax.tree.map(
                lambda x: x[:, : self.cfg.burn_in_length], experience
            )
            bi_obs, bi_done, bi_action, bi_reward = burn_in.first
            initial_actor_carry, (_, _) = self.actor_network.apply(
                jax.lax.stop_gradient(state.actor_params),
                observation=bi_obs,
                done=bi_done,
                action=bi_action,
                reward=bi_reward,
                initial_carry=initial_actor_carry,
            )
            initial_actor_carry = jax.lax.stop_gradient(initial_actor_carry)

            initial_critic_carry, _ = self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_params),
                observation=bi_obs,
                done=bi_done,
                action=bi_action,
                reward=bi_reward,
                initial_carry=initial_critic_carry,
            )
            initial_critic_carry = jax.lax.stop_gradient(initial_critic_carry)

            bi2_obs, bi2_done, bi2_action, bi2_reward = burn_in.second
            initial_target_critic_carry, _ = self.critic_network.apply(
                jax.lax.stop_gradient(state.critic_target_params),
                observation=bi2_obs,
                done=bi2_done,
                action=bi2_action,
                reward=bi2_reward,
                initial_carry=initial_target_critic_carry,
            )
            initial_target_critic_carry = jax.lax.stop_gradient(
                initial_target_critic_carry
            )

            experience = jax.tree.map(
                lambda x: x[:, self.cfg.burn_in_length :], experience
            )

        state = self._update_critic(
            critic_key,
            state,
            experience,
            initial_actor_carry,
            initial_critic_carry,
            initial_target_critic_carry,
        )
        state, actor_carry = self._update_actor(
            actor_key,
            state,
            experience,
            initial_actor_carry,
            initial_critic_carry,
        )
        state = self._update_alpha(
            alpha_key, state, experience, initial_actor_carry
        )

        return state

    def _update_step(self, state: SACState, key: Key):
        step_key, gradient_key = jax.random.split(key)

        step_keys = jax.random.split(step_key, self.cfg.train_frequency // self.cfg.num_envs)
        state, transitions = jax.lax.scan(
            partial(self._step, policy=self._stochastic_action),
            state,
            step_keys,
        )

        gradient_keys = jax.random.split(gradient_key, self.cfg.gradient_steps)
        state, _ = jax.lax.scan(
            lambda state, key: (self._update(key, state), None),
            state,
            gradient_keys,
        )
        lox.log({"training/step": state.step, "training/update_step": state.update_step})

        return state.replace(update_step=state.update_step + 1), None

    def warmup(self, key: Key, state: SACState, num_steps: int) -> SACState:
        step_keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            state,
            step_keys,
        )
        return state

    def train(self, key: Key, state: SACState, num_steps: int):
        keys = jax.random.split(key, num_steps // self.cfg.train_frequency)
        state, _ = jax.lax.scan(
            self._update_step,
            state,
            keys,
        )

        return state

    def evaluate(self, key: Key, state: SACState, num_steps: int) -> SACState:
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
        carry = self.actor_network.initialize_carry((self.cfg.num_envs, None))

        state = state.replace(
            timestep=timestep,
            env_state=env_state,
            actor_carry=carry,
        )

        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(
                self._step,
                policy=self._deterministic_action,
            ),
            state,
            step_keys,
        )

        return state
