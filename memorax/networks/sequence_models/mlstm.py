from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen import initializers
from flax.typing import Dtype

from memorax.networks.initializers import wang
from memorax.networks.layers import BlockDiagonalDense, MultiHeadLayerNorm
from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


@struct.dataclass
class mLSTMConfig:
    features: int
    hidden_dim: int
    num_heads: int = 4
    conv_kernel_size: int = 4
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32


@struct.dataclass
class mLSTMCarry:
    memory: Array
    log_forget: Array
    normalizer: Array
    max_log: Array


class mLSTMCell(MemoroidCellBase):
    config: mLSTMConfig

    def setup(self):
        if self.config.hidden_dim % self.config.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.config.hidden_dim}) must be divisible by "
                f"num_heads ({self.config.num_heads})."
            )

        self.up_projection = nn.Dense(
            2 * self.config.hidden_dim,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

        self.causal_conv = nn.Conv(
            features=self.config.hidden_dim,
            kernel_size=(self.config.conv_kernel_size,),
            padding=((self.config.conv_kernel_size - 1, 0),),
            feature_group_count=self.config.hidden_dim,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

        self.query = BlockDiagonalDense(
            self.config.hidden_dim,
            num_heads=self.config.num_heads,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )
        self.key = BlockDiagonalDense(
            self.config.hidden_dim,
            num_heads=self.config.num_heads,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )
        self.value = BlockDiagonalDense(
            self.config.hidden_dim,
            num_heads=self.config.num_heads,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

        self.i_gate = nn.Dense(
            self.config.num_heads,
            kernel_init=initializers.zeros_init(),
            bias_init=initializers.normal(stddev=0.1),
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )
        self.f_gate = nn.Dense(
            self.config.num_heads,
            kernel_init=initializers.zeros_init(),
            bias_init=lambda key, shape, dtype=jnp.float32: jnp.linspace(
                3.0, 6.0, shape[0], dtype=dtype
            ),
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

        self.learnable_skip = self.param(
            "learnable_skip",
            nn.initializers.ones,
            (self.config.hidden_dim,),
        )

        self.norm = MultiHeadLayerNorm(
            use_scale=True,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

        self.output_projection = nn.Dense(
            self.config.features,
            use_bias=False,
            kernel_init=wang(self.config.hidden_dim, num_blocks=1),
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

        self.drop = nn.Dropout(rate=self.config.dropout_rate)

    def _project(self, x: Array):
        B, T, _ = x.shape
        head_dim = self.config.hidden_dim // self.config.num_heads

        up = self.up_projection(x)
        x, z = jnp.split(up, 2, axis=-1)

        u = nn.silu(self.causal_conv(x))

        query = self.query(u).reshape(B, T, self.config.num_heads, head_dim)
        key = self.key(u).reshape(B, T, self.config.num_heads, head_dim)
        value = self.value(x).reshape(B, T, self.config.num_heads, head_dim)

        return query, key, value, z, u

    def __call__(self, x: Array, **kwargs) -> Carry:
        B, T, _ = x.shape
        head_dim = self.config.hidden_dim // self.config.num_heads

        query, key, value, _, _ = self._project(x)

        gate_input = jnp.concatenate([query, key, value], axis=-1).reshape(B, T, -1)

        i_gate = self.i_gate(gate_input).reshape(B, T, self.config.num_heads)
        f_gate = self.f_gate(gate_input).reshape(B, T, self.config.num_heads)

        log_f = -jax.nn.softplus(-f_gate)
        log_i = i_gate

        max_log = log_i[:, :, :, None, None]

        key = key / jnp.sqrt(head_dim)

        memory = jnp.einsum("...i,...j->...ij", key, value)
        normalizer = key[:, :, :, :, None]
        log_f = log_f[:, :, :, None, None]

        return mLSTMCarry(
            memory=memory, log_forget=log_f, normalizer=normalizer, max_log=max_log
        )

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        log_f = a.log_forget + b.log_forget

        m_a_decayed = a.max_log + b.log_forget
        m_combined = jnp.maximum(m_a_decayed, b.max_log)

        scale_a = jnp.exp(m_a_decayed - m_combined)
        scale_b = jnp.exp(b.max_log - m_combined)

        memory = scale_a * a.memory + scale_b * b.memory
        normalizer = scale_a * a.normalizer + scale_b * b.normalizer

        return mLSTMCarry(
            memory=memory, log_forget=log_f, normalizer=normalizer, max_log=m_combined
        )

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        B, T, _ = x.shape

        query, _, _, z, u = self._project(x)

        denominator = jnp.maximum(
            jnp.abs(jnp.einsum("...i,...i->...", query, h.normalizer[..., 0])),
            jnp.exp(-h.max_log[..., 0, 0]),
        )[..., None]
        h_tilde = jnp.einsum("...j,...jk->...k", query, h.memory) / (denominator + 1e-6)

        h_tilde = self.norm(h_tilde.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        h_tilde = h_tilde.reshape(B, T, self.config.hidden_dim)

        h_tilde = h_tilde + self.learnable_skip * u

        y = h_tilde * nn.silu(z)

        y = self.output_projection(y)

        y = self.drop(y, deterministic=not self.has_rng("dropout"))

        return y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        head_dim = self.config.hidden_dim // self.config.num_heads

        memory = jnp.zeros((*batch_dims, 1, self.config.num_heads, head_dim, head_dim))
        log_forget = jnp.zeros((*batch_dims, 1, self.config.num_heads, 1, 1))
        normalizer = jnp.zeros((*batch_dims, 1, self.config.num_heads, head_dim, 1))
        max_log = jnp.full((*batch_dims, 1, self.config.num_heads, 1, 1), -1e9)

        return mLSTMCarry(
            memory=memory, log_forget=log_forget, normalizer=normalizer, max_log=max_log
        )
