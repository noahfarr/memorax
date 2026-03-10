from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


def _initialize_nu_log(key, shape, r_min=0.0, r_max=1.0):
    u = jax.random.uniform(key, shape=shape)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def _initialize_theta_log(key, shape, max_phase=6.28):
    u = jax.random.uniform(key, shape=shape)
    return jnp.log(max_phase * u)


class RTUCell(MemoroidCellBase):
    hidden_dim: int
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    eps: float = 1e-8

    def setup(self):
        self.nu_log = self.param(
            "nu_log",
            partial(_initialize_nu_log, r_min=self.r_min, r_max=self.r_max),
            (self.hidden_dim,),
        )
        self.theta_log = self.param(
            "theta_log",
            partial(_initialize_theta_log, max_phase=self.max_phase),
            (self.hidden_dim,),
        )
        self.W_c1 = nn.Dense(self.hidden_dim, use_bias=False)
        self.W_c2 = nn.Dense(self.hidden_dim, use_bias=False)

    def _g_phi_norm(self):
        r = jnp.exp(-jnp.exp(self.nu_log))
        theta = jnp.exp(self.theta_log)
        g = r * jnp.cos(theta)
        phi = r * jnp.sin(theta)
        norm = jnp.sqrt(1 - r**2) + self.eps
        return g, phi, norm, r

    def __call__(self, x: Array, **kwargs) -> Carry:
        B, T, _ = x.shape
        g, phi, norm, _ = self._g_phi_norm()
        lam = jax.lax.complex(g, phi)
        B_input = jax.lax.complex(self.W_c1(x), self.W_c2(x))
        state = norm * B_input
        decay = jnp.broadcast_to(lam, (B, T, self.hidden_dim))
        return (state, decay)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        state_i, decay_i = a
        state_j, decay_j = b
        return (decay_j * state_i + state_j, decay_j * decay_i)

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        state, _ = h
        return nn.relu(jnp.concatenate([state.real, state.imag], axis=-1))

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        state = jnp.zeros((*batch_dims, 1, self.hidden_dim), dtype=jnp.complex64)
        decay = jnp.ones((*batch_dims, 1, self.hidden_dim), dtype=jnp.complex64)
        return (state, decay)

    def local_jacobian(self, carry, z, inputs, **kwargs):
        prev_state = carry[0]
        g, phi, norm, r = self._g_phi_norm()
        lam = jax.lax.complex(g, phi)

        B, T = inputs.shape[:2]
        decay_3d = jnp.broadcast_to(lam, (B, T, self.hidden_dim))

        d_norm_w_r = jnp.exp(self.nu_log) * r**2 / norm
        B_input = jax.lax.complex(self.W_c1(inputs), self.W_c2(inputs))

        dnu = (
            -jnp.exp(self.nu_log) * lam * prev_state
            + d_norm_w_r[None, None, :] * B_input
        )
        dtheta = 1j * jnp.exp(self.theta_log) * lam * prev_state

        dW_c1 = norm[None, None, :, None] * inputs[:, :, None, :]
        dW_c2 = 1j * norm[None, None, :, None] * inputs[:, :, None, :]

        return decay_3d, {
            "nu_log": dnu,
            "theta_log": dtheta,
            "W_c1/kernel": dW_c1,
            "W_c2/kernel": dW_c2,
        }

    def initialize_sensitivity(self, key, input_shape):
        *batch_dims, feat = input_shape
        H = self.hidden_dim
        z = lambda *s: jnp.zeros((*batch_dims, 1, *s), dtype=jnp.complex64)
        return {
            "nu_log": z(H),
            "theta_log": z(H),
            "W_c1/kernel": z(H, feat),
            "W_c2/kernel": z(H, feat),
        }
