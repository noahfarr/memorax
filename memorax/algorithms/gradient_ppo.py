from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from memorax.utils import Timestep, Transition, utils
from memorax.utils.axes import remove_feature_axis, remove_time_axis
from memorax.utils.typing import Array, Discrete, Environment, EnvParams, EnvState, Key, Carry, PyTree


@struct.dataclass(frozen=True)
class GradientPPOConfig:
    num_envs: int
    num_steps: int
    gamma: float
    gae_lambda: float
    num_minibatches: int
    update_epochs: int
    normalize_advantage: bool
    clip_coefficient: float
    clip_value_loss: bool
    entropy_coefficient: float
    regularization_coefficient: float
    truncation_length: int
    burn_in_length: int = 0

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@struct.dataclass(frozen=True)
class GradientPPOState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    actor_carry: Array
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: optax.OptState
    critic_carry: Array
    h_params: core.FrozenDict[str, Any]
    h_optimizer_state: optax.OptState
    h_carry: Array


@dataclass
class GradientPPO:
    cfg: GradientPPOConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module
    h_network: nn.Module
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    h_optimizer: optax.GradientTransformation

    def __post_init__(self):
        assert self.cfg.update_epochs >= 1, (
            f"update_epochs ({self.cfg.update_epochs}) must be >= 1"
        )
        assert self.cfg.num_steps % self.cfg.truncation_length == 0, (
            f"num_steps ({self.cfg.num_steps}) must be divisible by truncation_length ({self.cfg.truncation_length})"
        )
        num_truncations = self.cfg.num_envs * (self.cfg.num_steps // self.cfg.truncation_length)
        assert num_truncations % self.cfg.num_minibatches == 0, (
            f"num_envs * (num_steps // truncation_length) ({num_truncations}) must be divisible by num_minibatches ({self.cfg.num_minibatches})"
        )

    def _deterministic_action(
        self, key: Key, state: GradientPPOState
    ) -> tuple[GradientPPOState, Array, Array, None, dict]:
        obs, done, action, reward = state.timestep.to_sequence()
        (actor_carry, (probs, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=obs,
            done=done,
            action=action,
            reward=reward,
            initial_carry=state.actor_carry,
            mutable=["intermediates"],
        )

        action = (
            jnp.argmax(probs.logits, axis=-1)
            if isinstance(self.env.action_space(self.env_params), Discrete)
            else probs.mode()
        )
        log_prob = probs.log_prob(action)

        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)

        state = state.replace(
            actor_carry=actor_carry,
        )
        return state, action, log_prob, None, intermediates

    def _stochastic_action(
        self, key: Key, state: GradientPPOState
    ) -> tuple[GradientPPOState, Array, Array, Array, dict]:
        action_key, actor_torso_key, critic_torso_key = jax.random.split(key, 3)

        obs, done, ts_action, reward = state.timestep.to_sequence()
        (actor_carry, (probs, _)), intermediates = self.actor_network.apply(
            state.actor_params,
            observation=obs,
            done=done,
            action=ts_action,
            reward=reward,
            initial_carry=state.actor_carry,
            rngs={"torso": actor_torso_key},
            mutable=["intermediates"],
        )
        action, log_prob = probs.sample_and_log_prob(seed=action_key)

        critic_carry, (value, _) = self.critic_network.apply(
            state.critic_params,
            observation=obs,
            done=done,
            action=ts_action,
            reward=reward,
            initial_carry=state.critic_carry,
            rngs={"torso": critic_torso_key},
        )

        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)

        value = remove_time_axis(value)
        value = remove_feature_axis(value)

        state = state.replace(
            actor_carry=actor_carry,
            critic_carry=critic_carry,
        )
        return state, action, log_prob, value, intermediates

    def _step(
        self, state: GradientPPOState, key: Key, *, policy: Callable
    ) -> tuple[GradientPPOState, Transition]:
        step_carry = (state.actor_carry, state.critic_carry, state.h_carry)

        action_key, step_key = jax.random.split(key)
        state, action, log_prob, value, intermediates = policy(action_key, state)

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
        first = Timestep(
            obs=state.timestep.obs,
            action=jnp.where(
                jnp.expand_dims(state.timestep.done, axis=broadcast_dims),
                jnp.zeros_like(state.timestep.action),
                state.timestep.action,
            ),
            reward=jnp.where(state.timestep.done, 0, state.timestep.reward),
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
            carry=step_carry,
            aux={"log_prob": log_prob, "value": value},
        )

        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(
                obs=next_obs,
                action=action,
                reward=jnp.asarray(reward, dtype=jnp.float32),
                done=done,
            ),
            env_state=env_state,
        )
        return state, transition

    def _update_actor(
        self, key: Key, state: GradientPPOState, initial_actor_carry: Carry, transitions: Transition
    ) -> tuple[GradientPPOState, Array, tuple[Array, Array, Array]]:
        torso_key, dropout_key = jax.random.split(key)

        initial_actor_carry = utils.burn_in(
            self.actor_network, state.actor_params, transitions.first, initial_actor_carry, self.cfg.burn_in_length
        )
        transitions = jax.tree.map(lambda x: x[:, self.cfg.burn_in_length:], transitions)

        advantages = transitions.aux["advantages"]

        obs, done, action, reward = transitions.first

        def actor_loss_fn(params: PyTree):
            _, (probs, _) = self.actor_network.apply(
                params,
                observation=obs,
                done=done,
                action=action,
                reward=reward,
                initial_carry=initial_actor_carry,
                rngs={"torso": torso_key, "dropout": dropout_key},
            )
            log_probs = probs.log_prob(transitions.second.action)
            entropy = probs.entropy().mean()
            ratio = jnp.exp(log_probs - transitions.aux["log_prob"])
            approximate_kl = jnp.mean(transitions.aux["log_prob"] - log_probs)
            clip_fraction = jnp.mean(
                (jnp.abs(ratio - 1.0) > self.cfg.clip_coefficient).astype(jnp.float32)
            )

            actor_loss = -jnp.minimum(
                ratio * advantages,
                jnp.clip(
                    ratio,
                    1.0 - self.cfg.clip_coefficient,
                    1.0 + self.cfg.clip_coefficient,
                )
                * advantages,
            ).mean()
            return actor_loss - self.cfg.entropy_coefficient * entropy, (
                entropy.mean(),
                approximate_kl.mean(),
                clip_fraction.mean(),
            )

        (actor_loss, aux), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state.actor_params)
        lox.log({"actor/gradient_norm": optax.global_norm(actor_grads)})
        actor_updates, actor_optimizer_state = self.actor_optimizer.update(
            actor_grads, state.actor_optimizer_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, actor_updates)

        state = state.replace(
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
        )
        return state, actor_loss.mean(), aux

    def _compute_delta_lambda(self, critic_params: PyTree, transitions: Transition, initial_critic_carry: Carry):
        gamma = self.cfg.gamma
        first_obs, first_done, first_action, first_reward = transitions.first
        _, (values, _) = self.critic_network.apply(
            critic_params,
            observation=first_obs,
            done=first_done,
            action=first_action,
            reward=first_reward,
            initial_carry=initial_critic_carry,
        )
        values = remove_feature_axis(values)
        second_obs, second_done, second_action, second_reward = transitions.second
        _, (next_values, _) = self.critic_network.apply(
            critic_params,
            observation=second_obs,
            done=second_done,
            action=second_action,
            reward=second_reward,
            initial_carry=initial_critic_carry,
        )
        next_values = remove_feature_axis(next_values)
        deltas = (
            transitions.second.reward
            + gamma * (1 - transitions.second.done) * next_values
            - values
        )

        def scan_fn(delta_lambda_next: Array, delta_t: Array):
            delta_lambda = delta_t + gamma * self.cfg.gae_lambda * delta_lambda_next
            return delta_lambda, delta_lambda

        _, delta_lambda = jax.lax.scan(
            scan_fn,
            jnp.zeros_like(deltas[:, -1]),
            jnp.moveaxis(deltas, 1, 0),
            reverse=True,
        )
        delta_lambda = jnp.moveaxis(delta_lambda, 0, 1)
        return delta_lambda, values

    def _update_critic(self, key: Key, state: GradientPPOState, transitions: Transition, h_values: Array, initial_critic_carry: Carry):
        torso_key, dropout_key = jax.random.split(key)

        def critic_loss_fn(params: PyTree):
            delta_lambda, values = self._compute_delta_lambda(params, transitions, initial_critic_carry)
            critic_loss = (
                jax.lax.stop_gradient(h_values) * delta_lambda
                - jax.lax.stop_gradient(delta_lambda - h_values) * values
            ).mean()
            return critic_loss, (delta_lambda, values)

        (critic_loss, (delta_lambda, values)), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(state.critic_params)
        explained_variance = 1 - jnp.var(delta_lambda - values) / jnp.var(delta_lambda)
        lox.log({"critic/gradient_norm": optax.global_norm(critic_grads), "critic/explained_variance": explained_variance, "critic/value": values.mean()})
        critic_updates, critic_optimizer_state = self.critic_optimizer.update(
            critic_grads, state.critic_optimizer_state, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        state = state.replace(
            critic_params=critic_params, critic_optimizer_state=critic_optimizer_state
        )
        return state, critic_loss.mean(), delta_lambda

    def _update_h(self, key: Key, state: GradientPPOState, transitions: Transition, delta_lambda: Array, initial_h_carry: Carry):
        torso_key, dropout_key = jax.random.split(key)
        delta_lambda = jax.lax.stop_gradient(delta_lambda)

        obs, done, action, reward = transitions.first

        def h_loss_fn(params: PyTree):
            _, (h_values, _) = self.h_network.apply(
                params,
                observation=obs,
                done=done,
                action=action,
                reward=reward,
                initial_carry=initial_h_carry,
                rngs={"torso": torso_key, "dropout": dropout_key},
            )
            h_values = remove_feature_axis(h_values)
            h_loss = -(jax.lax.stop_gradient(delta_lambda - h_values) * h_values).mean()
            l2_reg = sum(
                jnp.sum(jnp.square(p))
                for p in jax.tree.leaves(params)
            )
            return h_loss + 0.5 * self.cfg.regularization_coefficient * l2_reg

        h_loss, h_grads = jax.value_and_grad(h_loss_fn)(state.h_params)
        h_updates, h_optimizer_state = self.h_optimizer.update(
            h_grads, state.h_optimizer_state, state.h_params
        )
        h_params = optax.apply_updates(state.h_params, h_updates)

        state = state.replace(
            h_params=h_params, h_optimizer_state=h_optimizer_state
        )
        return state, h_loss.mean()

    def _update_minibatch(
        self, state: GradientPPOState, xs: tuple
    ) -> tuple[GradientPPOState, tuple[Array, Array, tuple[Array, Array, Array]]]:
        minibatch, key = xs
        (
            initial_actor_carry,
            initial_critic_carry,
            initial_h_carry,
            transitions,
        ) = minibatch

        critic_key, h_key, actor_key = jax.random.split(key, 3)

        obs, done, action, reward = transitions.first
        _, (h_values, _) = self.h_network.apply(
            state.h_params,
            observation=obs,
            done=done,
            action=action,
            reward=reward,
            initial_carry=initial_h_carry,
        )
        h_values = remove_feature_axis(h_values)
        h_values = jax.lax.stop_gradient(h_values)

        state, critic_loss, delta_lambda = self._update_critic(
            critic_key, state, transitions, h_values, initial_critic_carry
        )

        state, h_loss = self._update_h(
            h_key, state, transitions, delta_lambda, initial_h_carry
        )

        delta_lambda = jax.lax.stop_gradient(delta_lambda)
        if self.cfg.normalize_advantage:
            delta_lambda = (
                (delta_lambda - delta_lambda.mean())
                / (delta_lambda.std() + 1e-8)
            )
        transitions = transitions.replace(aux={**transitions.aux, "advantages": delta_lambda})
        state, actor_loss, aux = self._update_actor(
            actor_key, state, initial_actor_carry, transitions
        )

        return state, (actor_loss, critic_loss, aux)

    def _update_epoch(self, carry: tuple, key: Key) -> tuple:
        (
            state,
            initial_actor_carry,
            initial_critic_carry,
            initial_h_carry,
            transitions,
        ) = carry

        permutation_key, minibatch_key = jax.random.split(key)

        def shuffle(batch: PyTree):
            shuffle_time_axis = (
                initial_actor_carry is None or initial_critic_carry is None
            )
            num_permutations = self.cfg.num_envs * (self.cfg.num_steps // self.cfg.truncation_length)
            if shuffle_time_axis:
                batch = (
                    initial_actor_carry,
                    initial_critic_carry,
                    initial_h_carry,
                    jax.tree.map(
                        lambda x: x.reshape(-1, *x.shape[2:]),
                        transitions,
                    ),
                )
                num_permutations *= self.cfg.num_steps // self.cfg.truncation_length

            permutation = jax.random.permutation(permutation_key, num_permutations)

            minibatches = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0).reshape(
                    self.cfg.num_minibatches, -1, *x.shape[1:]
                ),
                batch,
            )
            return minibatches

        minibatches = shuffle(
            (
                initial_actor_carry,
                initial_critic_carry,
                initial_h_carry,
                transitions,
            )
        )
        minibatch_keys = jax.random.split(minibatch_key, self.cfg.num_minibatches)

        state, (actor_loss, critic_loss, (entropy, approximate_kl, clip_fraction)) = (
            jax.lax.scan(
                self._update_minibatch,
                state,
                (minibatches, minibatch_keys),
            )
        )

        metrics = jax.tree.map(
            lambda x: x.mean(), (actor_loss, critic_loss, entropy, approximate_kl, clip_fraction)
        )

        return (
            state,
            initial_actor_carry,
            initial_critic_carry,
            initial_h_carry,
            transitions,
        ), metrics

    def _update_step(self, state: GradientPPOState, key: Key) -> tuple[GradientPPOState, None]:
        step_key, epoch_key = jax.random.split(key)

        step_keys = jax.random.split(step_key, self.cfg.num_steps)
        state, transitions = jax.lax.scan(
            partial(self._step, policy=self._stochastic_action),
            state,
            step_keys,
        )

        transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), transitions)

        transitions = jax.tree.map(
            lambda x: x.reshape(
                self.cfg.num_envs, -1, self.cfg.truncation_length, *x.shape[2:]
            ),
            transitions,
        )

        initial_actor_carry, initial_critic_carry, initial_h_carry = jax.tree.map(
            lambda x: x[:, :, 0], transitions.carry
        )

        transitions = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]),
            transitions,
        )
        initial_actor_carry, initial_critic_carry, initial_h_carry = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]),
            (initial_actor_carry, initial_critic_carry, initial_h_carry),
        )

        epoch_keys = jax.random.split(epoch_key, self.cfg.update_epochs)
        (state, *_, transitions), metrics = jax.lax.scan(
            self._update_epoch,
            (
                state,
                initial_actor_carry,
                initial_critic_carry,
                initial_h_carry,
                transitions,
            ),
            epoch_keys,
        )

        actor_loss, critic_loss, entropy, approximate_kl, clip_fraction = jax.tree.map(
            lambda x: x.mean(), metrics
        )
        lox.log({
            "actor/loss": actor_loss,
            "critic/loss": critic_loss,
            "actor/entropy": entropy,
            "actor/approximate_kl": approximate_kl,
            "actor/clip_fraction": clip_fraction,
            "training/step": state.step,
            "training/update_step": state.update_step,
        })

        return state.replace(update_step=state.update_step + 1), None

    def init(self, key: Key) -> GradientPPOState:
        (
            env_key,
            actor_key,
            actor_torso_key,
            actor_dropout_key,
            critic_key,
            critic_torso_key,
            critic_dropout_key,
            h_key,
            h_torso_key,
            h_dropout_key,
        ) = jax.random.split(key, 10)

        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()
        actor_carry = self.actor_network.initialize_carry((self.cfg.num_envs, None))
        critic_carry = self.critic_network.initialize_carry((self.cfg.num_envs, None))
        h_carry = self.h_network.initialize_carry((self.cfg.num_envs, None))

        ts_obs, ts_done, ts_action, ts_reward = timestep
        actor_params = self.actor_network.init(
            {
                "params": actor_key,
                "torso": actor_torso_key,
                "dropout": actor_dropout_key,
            },
            observation=ts_obs,
            done=ts_done,
            action=ts_action,
            reward=ts_reward,
            initial_carry=actor_carry,
        )
        critic_params = self.critic_network.init(
            {
                "params": critic_key,
                "torso": critic_torso_key,
                "dropout": critic_dropout_key,
            },
            observation=ts_obs,
            done=ts_done,
            action=ts_action,
            reward=ts_reward,
            initial_carry=critic_carry,
        )
        h_params = self.h_network.init(
            {
                "params": h_key,
                "torso": h_torso_key,
                "dropout": h_dropout_key,
            },
            observation=ts_obs,
            done=ts_done,
            action=ts_action,
            reward=ts_reward,
            initial_carry=h_carry,
        )

        actor_optimizer_state = self.actor_optimizer.init(actor_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)
        h_optimizer_state = self.h_optimizer.init(h_params)

        return GradientPPOState(
            step=0,
            update_step=0,
            timestep=timestep.from_sequence(),
            actor_carry=actor_carry,
            critic_carry=critic_carry,
            h_carry=h_carry,
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            h_params=h_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            h_optimizer_state=h_optimizer_state,
        )

    def warmup(
        self, key: Key, state: GradientPPOState, num_steps: int
    ) -> GradientPPOState:
        return state

    def train(self, key: Key, state: GradientPPOState, num_steps: int) -> GradientPPOState:
        num_outer_steps = num_steps // (self.cfg.num_envs * self.cfg.num_steps)
        keys = jax.random.split(key, num_outer_steps)
        state, _ = jax.lax.scan(
            self._update_step,
            state,
            keys,
        )

        return state

    def evaluate(
        self, key: Key, state: GradientPPOState, num_steps: int
    ) -> GradientPPOState:
        reset_key, eval_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        initial_actor_carry = self.actor_network.initialize_carry(
            (self.cfg.num_envs, None)
        )
        initial_critic_carry = self.critic_network.initialize_carry(
            (self.cfg.num_envs, None)
        )
        state = state.replace(
            timestep=timestep,
            actor_carry=initial_actor_carry,
            critic_carry=initial_critic_carry,
            env_state=env_state,
        )

        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._deterministic_action),
            state,
            step_keys,
        )

        return state
