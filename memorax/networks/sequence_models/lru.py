from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from flax.typing import Dtype

from memorax.utils.typing import Array, Carry, Key

from .memoroid import MemoroidCellBase


def _nu_init(key, shape, r_min, r_max, dtype=jnp.float32):
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def _theta_init(key, shape, max_phase, dtype=jnp.float32):
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def _gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


def _matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


@struct.dataclass
class LRUConfig:
    features: int
    hidden_dim: int
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32


@struct.dataclass
class LRUCarry:
    state: Array
    decay: Array


class LRUCell(MemoroidCellBase):
    config: LRUConfig

    def setup(self):
        self.theta_log = self.param(
            "theta_log",
            partial(_theta_init, max_phase=self.config.max_phase),
            (self.config.hidden_dim,),
        )
        self.nu_log = self.param(
            "nu_log",
            partial(_nu_init, r_min=self.config.r_min, r_max=self.config.r_max),
            (self.config.hidden_dim,),
        )
        self.gamma_log = self.param(
            "gamma_log", _gamma_log_init, (self.nu_log, self.theta_log)
        )

        self.B_real = self.param(
            "B_real",
            partial(_matrix_init, normalization=jnp.sqrt(2 * self.config.features)),
            (self.config.hidden_dim, self.config.features),
        )
        self.B_imag = self.param(
            "B_imag",
            partial(_matrix_init, normalization=jnp.sqrt(2 * self.config.features)),
            (self.config.hidden_dim, self.config.features),
        )
        self.C_real = self.param(
            "C_real",
            partial(_matrix_init, normalization=jnp.sqrt(self.config.hidden_dim)),
            (self.config.features, self.config.hidden_dim),
        )
        self.C_imag = self.param(
            "C_imag",
            partial(_matrix_init, normalization=jnp.sqrt(self.config.hidden_dim)),
            (self.config.features, self.config.hidden_dim),
        )
        self.D = self.param("D", _matrix_init, (self.config.features,))

    def __call__(self, x: Array, **kwargs) -> Carry:
        B, T, _ = x.shape

        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_real + 1j * self.B_imag) * jnp.exp(self.gamma_log)[:, None]

        decay = jnp.broadcast_to(diag_lambda, (B, T, self.config.hidden_dim))

        state = jnp.einsum('ij,btj->bti', B_norm, x)

        return LRUCarry(state=state, decay=decay)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        return LRUCarry(
            state=b.decay * a.state + b.state,
            decay=b.decay * a.decay,
        )

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        C = jax.lax.complex(self.C_real, self.C_imag)
        y = jnp.einsum('ij,btj->bti', C, h.state).real + self.D * x
        return y

    def initialize_carry(self, key: jax.Array, input_shape: tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        state = jnp.zeros((*batch_dims, 1, self.config.hidden_dim), dtype=jnp.complex64)
        decay = jnp.ones((*batch_dims, 1, self.config.hidden_dim), dtype=jnp.complex64)
        return LRUCarry(state=state, decay=decay)

    def inject_phantom(self, carry: Carry, phantom: Array) -> Carry:
        return carry.replace(state=jax.lax.stop_gradient(carry.state) + phantom)

    def local_jacobian(self, carry, z, inputs, **kwargs) -> tuple[Array, dict]:
        lam = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        gamma_exp = jnp.exp(self.gamma_log)

        B, T = inputs.shape[:2]
        decay_3d = jnp.broadcast_to(lam, (B, T, self.config.hidden_dim))

        return decay_3d, {
            "nu_log": -jnp.exp(self.nu_log) * lam * carry.state,
            "theta_log": 1j * jnp.exp(self.theta_log) * lam * carry.state,
            "gamma_log": z.state,
            "B_real": jnp.einsum('h,btf->bthf', gamma_exp, inputs),
            "B_imag": 1j * jnp.einsum('h,btf->bthf', gamma_exp, inputs),
        }

    def initialize_sensitivity(self, key: Key, input_shape: tuple) -> dict:
        *batch_dims, _ = input_shape
        H = self.config.hidden_dim
        z = lambda *s: jnp.zeros((*batch_dims, 1, *s), dtype=jnp.complex64)
        sensitivity = {
            "nu_log": z(H),
            "theta_log": z(H),
            "gamma_log": z(H),
            "B_real": z(H, self.config.features),
            "B_imag": z(H, self.config.features),
        }
        return sensitivity
