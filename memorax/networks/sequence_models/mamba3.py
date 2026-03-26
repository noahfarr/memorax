from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.typing import Dtype, Initializer

from memorax.networks.initializers import inverse_softplus
from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


@struct.dataclass
class Mamba3Config:
    features: int
    num_heads: int = 8
    head_dim: int = 16
    state_dim: int = 16
    kernel_init: Initializer = struct.field(
        pytree_node=False, default=nn.initializers.lecun_normal()
    )
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32


@struct.dataclass
class Mamba3Carry:
    state: Array
    state_input: Array
    decay: Array
    beta: Array


class Mamba3Cell(MemoroidCellBase):
    config: Mamba3Config

    def setup(self):
        hidden_dim = self.config.num_heads * self.config.head_dim
        state_projection_dim = self.config.num_heads * self.config.state_dim
        complex_state_dim = self.config.state_dim // 2

        self.D = self.param("D", nn.initializers.ones, (self.config.num_heads,))
        self.dt_bias = self.param(
            "dt_bias", inverse_softplus(), (self.config.num_heads,)
        )
        self.B_bias = self.param(
            "B_bias",
            nn.initializers.ones,
            (self.config.num_heads, self.config.state_dim),
        )
        self.C_bias = self.param(
            "C_bias",
            nn.initializers.ones,
            (self.config.num_heads, self.config.state_dim),
        )

        projection = partial(
            nn.Dense,
            kernel_init=self.config.kernel_init,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

        self.A = projection(self.config.num_heads)
        self.input_projection = projection(hidden_dim * 2)
        self.B = projection(state_projection_dim)
        self.C = projection(state_projection_dim)
        self.dt = projection(self.config.num_heads)
        self.theta = projection(self.config.num_heads * complex_state_dim)
        self.lam = projection(self.config.num_heads)

        self.B_norm = nn.RMSNorm(self.config.state_dim)
        self.C_norm = nn.RMSNorm(self.config.state_dim)
        self.output_projection = nn.Dense(
            self.config.features,
            kernel_init=self.config.kernel_init,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

    def _project(self, x: Array):
        batch_size, sequence_length, _ = x.shape
        complex_state_dim = self.config.state_dim // 2

        hidden, gate = jnp.split(self.input_projection(x), 2, axis=-1)
        hidden = hidden.reshape(
            batch_size, sequence_length, self.config.num_heads, self.config.head_dim
        )

        B = self.B(x).reshape(
            batch_size, sequence_length, self.config.num_heads, self.config.state_dim
        )
        C = self.C(x).reshape(
            batch_size, sequence_length, self.config.num_heads, self.config.state_dim
        )

        B = self.B_norm(B) + self.B_bias
        C = self.C_norm(C) + self.C_bias

        B_complex = jax.lax.complex(
            B[..., :complex_state_dim],
            B[..., complex_state_dim:],
        )
        C_complex = jax.lax.complex(
            C[..., :complex_state_dim],
            C[..., complex_state_dim:],
        )

        theta = (nn.sigmoid(2 * self.theta(x)) * 2 - 1) * jnp.pi
        theta = theta.reshape(
            batch_size, sequence_length, self.config.num_heads, complex_state_dim
        )

        dt = nn.softplus(self.dt(x) + self.dt_bias)
        lambda_value = nn.sigmoid(self.lam(x))

        A = -nn.softplus(self.A(x))
        decay_complex = jnp.exp(
            jax.lax.complex((dt * A)[..., None], dt[..., None] * theta)
        )

        beta_complex = ((1 - lambda_value) * dt)[..., None] * decay_complex
        gamma = lambda_value * dt

        return hidden, B_complex, C_complex, gate, decay_complex, beta_complex, gamma

    def __call__(self, x: Array, **kwargs) -> Carry:
        hidden, B_complex, _, _, decay_complex, beta_complex, gamma = self._project(x)

        state_input = jnp.einsum(
            "bthn,bthd->bthnd", B_complex, hidden.astype(jnp.complex64)
        )
        state_contribution = jnp.einsum("bth,bthnd->bthnd", gamma, state_input)

        return Mamba3Carry(
            state=state_contribution,
            state_input=state_input,
            decay=decay_complex[..., None],
            beta=beta_complex[..., None],
        )

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        return Mamba3Carry(
            state=b.decay * a.state + b.beta * a.state_input + b.state,
            state_input=b.state_input,
            decay=b.decay * a.decay,
            beta=b.decay * a.beta,
        )

    def read(self, carry: Carry, x: Array, **kwargs) -> Array:
        batch_size, sequence_length, _ = x.shape

        hidden, _, C_complex, gate, _, _, _ = self._project(x)

        output = jnp.einsum("bthn,bthnd->bthd", jnp.conj(C_complex), carry.state).real
        output = output + jnp.einsum("h,bthd->bthd", self.D, hidden)
        output = output.reshape(
            batch_size, sequence_length, self.config.num_heads * self.config.head_dim
        )
        output = output * nn.silu(gate)

        return self.output_projection(output)

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        complex_state_dim = self.config.state_dim // 2
        state = jnp.zeros(
            (
                *batch_dims,
                1,
                self.config.num_heads,
                complex_state_dim,
                self.config.head_dim,
            ),
            dtype=jnp.complex64,
        )
        state_input = jnp.zeros_like(state)
        decay = jnp.ones(
            (*batch_dims, 1, self.config.num_heads, complex_state_dim, 1),
            dtype=jnp.complex64,
        )
        beta = jnp.zeros_like(decay)
        return Mamba3Carry(state=state, state_input=state_input, decay=decay, beta=beta)
