from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.typing import Dtype
from jax.nn.initializers import lecun_normal, normal

from memorax.utils.typing import Array, Carry

from memorax.networks.hippo import discretize_bilinear, discretize_zoh, make_dplr_hippo
from memorax.networks.initializers import init_cv, init_v_inv_b, log_step, truncated_standard_normal

from .memoroid import MemoroidCellBase


def _c_init_complex_normal(key, shape):
    return normal(stddev=0.5**0.5)(key, shape)


def _c_init_lecun(key, shape, v):
    return init_cv(lecun_normal(), key, shape, v)


def _c_init_truncated(key, shape, v):
    return init_cv(truncated_standard_normal, key, shape, v)


@struct.dataclass
class S5Config:
    features: int
    hidden_dim: int
    c_init: str = "truncated_standard_normal"
    discretization: str = "zoh"
    dt_min: float = 0.001
    dt_max: float = 0.1
    clip_eigens: bool = False
    step_rescale: float = 1.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32


@struct.dataclass
class S5Carry:
    state: Array
    decay: Array


class S5Cell(MemoroidCellBase):
    config: S5Config

    def setup(self):
        lam, _, _, v, _ = make_dplr_hippo(self.config.hidden_dim)
        lambda_real_init = jnp.asarray(lam.real, self.config.param_dtype)
        lambda_imag_init = jnp.asarray(lam.imag, self.config.param_dtype)
        self._v = v
        self._v_inv = v.conj().T

        self.lambda_real = self.param(
            "lambda_real",
            lambda rng, shape: lambda_real_init,
            (self.config.hidden_dim,),
        )
        self.lambda_imag = self.param(
            "lambda_imag",
            lambda rng, shape: lambda_imag_init,
            (self.config.hidden_dim,),
        )
        self.B = self.param(
            "b",
            lambda rng, shape: init_v_inv_b(
                lecun_normal(), rng, (self.config.hidden_dim, self.config.features), self._v_inv
            ),
            (self.config.hidden_dim, self.config.features, 2),
        )

        match self.config.c_init:
            case "complex_normal":
                self.C = self.param(
                    "c",
                    _c_init_complex_normal,
                    (self.config.features, self.config.hidden_dim, 2),
                )
            case "lecun_normal":
                self.C = self.param(
                    "c",
                    partial(_c_init_lecun, v=self._v),
                    (self.config.features, self.config.hidden_dim, 2),
                )
            case "truncated_standard_normal":
                self.C = self.param(
                    "c",
                    partial(_c_init_truncated, v=self._v),
                    (self.config.features, self.config.hidden_dim, 2),
                )
            case _:
                raise ValueError(f"Invalid c_init: {self.config.c_init}")

        self.d = self.param("d", normal(stddev=1.0), (self.config.features,))
        self.log_step = self.param(
            "log_step", log_step(self.config.dt_min, self.config.dt_max), (self.config.hidden_dim,)
        )

    def _discretized_params(self):
        lambda_real = self.lambda_real
        lambda_imag = self.lambda_imag

        if self.config.clip_eigens:
            lambda_real = jnp.minimum(lambda_real, -1e-4)

        lam = jax.lax.complex(lambda_real, lambda_imag)
        b_tilde = jax.lax.complex(self.B[..., 0], self.B[..., 1])
        c_tilde = jax.lax.complex(self.C[..., 0], self.C[..., 1])
        step = self.config.step_rescale * jnp.exp(self.log_step[:, 0].astype(jnp.float32))

        match self.config.discretization:
            case "zoh":
                lambda_bar, b_bar = discretize_zoh(
                    lam.astype(jnp.complex64),
                    b_tilde.astype(jnp.complex64),
                    step.astype(jnp.complex64),
                )
            case "bilinear":
                lambda_bar, b_bar = discretize_bilinear(
                    lam.astype(jnp.complex64),
                    b_tilde.astype(jnp.complex64),
                    step.astype(jnp.complex64),
                )
            case _:
                raise ValueError(f"Invalid discretization: {self.config.discretization}")

        d = self.d.astype(self.config.dtype or self.config.param_dtype)
        return lambda_bar, b_bar, c_tilde, d, lam, b_tilde, step

    def __call__(self, x: Array, **kwargs) -> Carry:
        B, T, _ = x.shape
        lambda_bar, b_bar, _, _, _, _, _ = self._discretized_params()

        decay = jnp.broadcast_to(lambda_bar, (B, T, self.config.hidden_dim))

        state = jnp.einsum('ij,btj->bti', b_bar, x)

        return S5Carry(state=state, decay=decay)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        return S5Carry(
            state=b.decay * a.state + b.state,
            decay=b.decay * a.decay,
        )

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        _, _, c_tilde, d, _, _, _ = self._discretized_params()

        y = jnp.einsum('ij,btj->bti', c_tilde, h.state).real + d * x
        return y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        state = jnp.zeros((*batch_dims, 1, self.config.hidden_dim), dtype=jnp.complex64)
        decay = jnp.ones((*batch_dims, 1, self.config.hidden_dim), dtype=jnp.complex64)
        return S5Carry(state=state, decay=decay)
