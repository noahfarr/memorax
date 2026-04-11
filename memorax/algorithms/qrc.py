from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from memorax.utils import Timestep
from memorax.utils.axes import add_feature_axis, remove_feature_axis, remove_time_axis
from memorax.utils.typing import (
    Array,
    Carry,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class QRCConfig:
    num_envs: int
    gamma: float
    lamda: float
    gradient_correction: bool
    reg_coeff: float


@struct.dataclass(frozen=True)
class QRCState:
    step: int
    update_step: int
    timestep: Timestep
    q_carry: Carry
    h_carry: Carry
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    h_params: core.FrozenDict[str, Any]
    q_optimizer_state: optax.OptState
    h_optimizer_state: optax.OptState
    h_trace: Array
    q_traces: PyTree
    h_traces: PyTree


@dataclass
class QRC:
    cfg: QRCConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    h_network: nn.Module
    q_optimizer: optax.GradientTransformation
    h_optimizer: optax.GradientTransformation
    epsilon_schedule: optax.Schedule

    def _greedy_action(
        self, key: Key, state: QRCState
    ) -> tuple[QRCState, Array, Array, dict]:
        (carry, (q_values, _)), intermediates = self.q_network.apply(
            state.q_params,
            *state.timestep.to_sequence(),
            initial_carry=state.q_carry,
            rngs={"torso": key},
            mutable=["intermediates"],
        )
        action = remove_time_axis(jnp.argmax(q_values, axis=-1))
        return (
            state.replace(q_carry=carry),
            action,
            jnp.zeros(self.cfg.num_envs, dtype=jnp.bool_),
            intermediates,
        )

    def _random_action(
        self, key: Key, state: QRCState
    ) -> tuple[QRCState, Array, Array, dict]:
        keys = jax.random.split(key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(keys)
        return state, action, jnp.ones(self.cfg.num_envs, dtype=jnp.bool_), {}

    def _epsilon_greedy_action(
        self, key: Key, state: QRCState
    ) -> tuple[QRCState, Array, Array, dict]:
        random_key, greedy_key, sample_key = jax.random.split(key, 3)
        _, sampled_action, _, _ = self._random_action(random_key, state)
        state, greedy_action, _, intermediates = self._greedy_action(greedy_key, state)
        epsilon = self.epsilon_schedule(state.step)
        random_action = jax.random.uniform(sample_key, greedy_action.shape) < epsilon
        action = jnp.where(random_action, sampled_action, greedy_action)
        return state, action, random_action, intermediates

    def _update(
        self,
        key: Key,
        state: QRCState,
        action: Array,
        next_obs: Array,
        reward: Array,
        done: Array,
        random_action: Array,
        q_carry: Carry,
        h_carry: Carry,
    ) -> QRCState:
        second = Timestep(obs=next_obs, done=done, action=action, reward=reward)

        def q_loss_fn(params):
            _, (q_values, _) = self.q_network.apply(
                params,
                *state.timestep.to_sequence(),
                initial_carry=q_carry,
                rngs={"torso": key},
            )
            return remove_feature_axis(
                jnp.take_along_axis(q_values[:, 0], add_feature_axis(action), axis=-1)
            )

        def td_loss_fn(params):
            _, (q_values, _) = self.q_network.apply(
                params,
                *state.timestep.to_sequence(),
                initial_carry=q_carry,
                rngs={"torso": key},
            )
            q_value = remove_feature_axis(
                jnp.take_along_axis(q_values[:, 0], add_feature_axis(action), axis=-1)
            )
            _, (next_q_values, _) = self.q_network.apply(
                params,
                *second.to_sequence(),
                initial_carry=q_carry,
                rngs={"torso": key},
            )
            next_q_value = next_q_values[:, 0].max(axis=-1)
            return (
                second.reward
                + self.cfg.gamma * next_q_value * (1.0 - second.done)
                - q_value
            )

        def h_loss_fn(params):
            _, (h_values, _) = self.h_network.apply(
                params,
                *state.timestep.to_sequence(),
                initial_carry=h_carry,
                rngs={"torso": key},
            )
            return remove_feature_axis(
                jnp.take_along_axis(h_values[:, 0], add_feature_axis(action), axis=-1)
            )

        q_grads = jax.jacobian(q_loss_fn)(state.q_params)
        td_errors = td_loss_fn(state.q_params)
        td_grads = jax.jacobian(td_loss_fn)(state.q_params)
        correction = h_loss_fn(state.h_params)
        h_grads = jax.jacobian(h_loss_fn)(state.h_params)

        h_trace = self.cfg.gamma * self.cfg.lamda * state.h_trace + correction
        q_traces = jax.tree.map(
            lambda e, g: self.cfg.gamma * self.cfg.lamda * e + g,
            state.q_traces,
            q_grads,
        )
        h_traces = jax.tree.map(
            lambda e, g: self.cfg.gamma * self.cfg.lamda * e + g,
            state.h_traces,
            h_grads,
        )

        def broadcast(v, x):
            return v[(slice(None),) + (None,) * (x.ndim - 1)]

        q_updates = jax.tree.map(
            lambda td_g: -broadcast(h_trace, td_g) * td_g, td_grads
        )
        if self.cfg.gradient_correction:
            q_updates = jax.tree.map(
                lambda upd, eq, qg: upd
                + broadcast(td_errors, eq) * eq
                - broadcast(correction, qg) * qg,
                q_updates,
                q_traces,
                q_grads,
            )

        h_updates = jax.tree.map(
            lambda eh, hg, p: broadcast(td_errors, eh) * eh
            - broadcast(correction, hg) * hg
            - self.cfg.reg_coeff * p[None],
            h_traces,
            h_grads,
            state.h_params,
        )

        q_grads = jax.tree.map(lambda x: -x.mean(axis=0), q_updates)
        h_grads = jax.tree.map(lambda x: -x.mean(axis=0), h_updates)

        q_param_updates, q_optimizer_state = self.q_optimizer.update(
            q_grads,
            state.q_optimizer_state,
            state.q_params,
        )
        h_param_updates, h_optimizer_state = self.h_optimizer.update(
            h_grads,
            state.h_optimizer_state,
            state.h_params,
        )
        q_params = optax.apply_updates(state.q_params, q_param_updates)
        h_params = optax.apply_updates(state.h_params, h_param_updates)

        reset = done | random_action

        def reset_trace(trace):
            return jnp.where(
                reset[(slice(None),) + (None,) * (trace.ndim - 1)],
                jnp.zeros_like(trace),
                trace,
            )

        h_trace = jnp.where(reset, jnp.zeros_like(h_trace), h_trace)
        q_traces = jax.tree.map(reset_trace, q_traces)
        h_traces = jax.tree.map(reset_trace, h_traces)

        q_value = q_loss_fn(state.q_params)
        lox.log(
            {
                "q_network/q_value": q_value.mean(),
                "q_network/td_error": td_errors.mean(),
                "q_network/h_trace": h_trace.mean(),
                "q_network/gradient_norm": optax.global_norm(q_grads),
                "h_network/correction": correction.mean(),
                "h_network/gradient_norm": optax.global_norm(h_grads),
                "training/epsilon": self.epsilon_schedule(state.step),
                "training/step": state.step,
                "training/update_step": state.update_step,
            }
        )

        return state.replace(
            q_params=q_params,
            h_params=h_params,
            q_optimizer_state=q_optimizer_state,
            h_optimizer_state=h_optimizer_state,
            h_trace=h_trace,
            q_traces=q_traces,
            h_traces=h_traces,
        )

    def _step(
        self, state: QRCState, key: Key, *, policy: Callable
    ) -> tuple[QRCState, None]:
        action_key, step_key, update_key = jax.random.split(key, 3)

        q_carry = state.q_carry
        h_carry = state.h_carry

        state, action, random_action, intermediates = policy(action_key, state)
        intermediates = jax.tree.map(
            lambda x: jnp.mean(jnp.stack(x)),
            intermediates.get("intermediates", {}),
            is_leaf=lambda x: isinstance(x, tuple),
        )

        step_keys = jax.random.split(step_key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)

        lox.log({"intermediates": intermediates, "info": info})

        reward = jnp.asarray(reward, dtype=jnp.float32)
        state = self._update(
            update_key,
            state,
            action,
            next_obs,
            reward,
            done,
            random_action,
            q_carry,
            h_carry,
        )

        h_carry, _ = self.h_network.apply(
            state.h_params,
            *state.timestep.to_sequence(),
            initial_carry=state.h_carry,
            rngs={"torso": action_key},
        )

        return (
            state.replace(
                step=state.step + self.cfg.num_envs,
                update_step=state.update_step + 1,
                timestep=Timestep(
                    obs=next_obs, action=action, reward=reward, done=done
                ),
                env_state=env_state,
                h_carry=h_carry,
            ),
            None,
        )

    def init(self, key: Key) -> QRCState:
        env_key, q_key, h_key, torso_key = jax.random.split(key, 4)
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

        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        q_carry = self.q_network.initialize_carry((self.cfg.num_envs, None))
        h_carry = self.h_network.initialize_carry((self.cfg.num_envs, None))

        q_params = self.q_network.init(
            {"params": q_key, "torso": torso_key},
            *timestep.to_sequence(),
            initial_carry=q_carry,
        )
        h_params = self.h_network.init(
            {"params": h_key, "torso": torso_key},
            *timestep.to_sequence(),
            initial_carry=h_carry,
        )

        q_optimizer_state = self.q_optimizer.init(q_params)
        h_optimizer_state = self.h_optimizer.init(h_params)

        return QRCState(
            step=0,
            update_step=0,
            timestep=timestep,
            q_carry=q_carry,
            h_carry=h_carry,
            env_state=env_state,
            q_params=q_params,
            h_params=h_params,
            q_optimizer_state=q_optimizer_state,
            h_optimizer_state=h_optimizer_state,
            h_trace=jnp.zeros((self.cfg.num_envs,)),
            q_traces=jax.tree.map(
                lambda p: jnp.zeros((self.cfg.num_envs, *p.shape)), q_params
            ),
            h_traces=jax.tree.map(
                lambda p: jnp.zeros((self.cfg.num_envs, *p.shape)), h_params
            ),
        )

    def warmup(self, key: Key, state: QRCState, num_steps: int) -> QRCState:
        step_keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            state,
            step_keys,
        )
        return state

    def train(self, key: Key, state: QRCState, num_steps: int) -> QRCState:
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            state,
            keys,
        )
        return state

    def evaluate(self, key: Key, state: QRCState, num_steps: int) -> QRCState:
        reset_key, eval_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        q_carry = self.q_network.initialize_carry((self.cfg.num_envs, None))
        h_carry = self.h_network.initialize_carry((self.cfg.num_envs, None))
        state = state.replace(
            timestep=Timestep(obs=obs, action=action, reward=reward, done=done),
            q_carry=q_carry,
            h_carry=h_carry,
            env_state=env_state,
        )
        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._greedy_action),
            state,
            step_keys,
        )
        return state
