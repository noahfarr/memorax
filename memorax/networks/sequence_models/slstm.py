from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen import initializers
from flax.linen.recurrent import RNNCellBase
from flax.typing import Dtype
from jax import random

from memorax.utils.typing import Array

from memorax.networks.layers import BlockDiagonalDense, CausalConv1d, MultiHeadLayerNorm
from memorax.networks.initializers import powerlaw
from memorax.utils.axes import add_time_axis, remove_time_axis


@struct.dataclass
class sLSTMConfig:
    features: int
    hidden_dim: int
    num_heads: int = 4
    use_causal_conv: bool = True
    conv_kernel_size: int = 4
    eps: float = 1e-6
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32


@struct.dataclass
class sLSTMCarry:
    cell: Array
    normalizer: Array
    max_log: Array
    hidden: Array
    buffer: Array


class sLSTMCell(RNNCellBase):
    config: sLSTMConfig

    def setup(self):
        head_dim = self.config.hidden_dim // self.config.num_heads
        if self.config.hidden_dim % self.config.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.config.hidden_dim}) must be divisible by num_heads ({self.config.num_heads})."
            )

        self.input_projection = nn.Dense(
            features=self.config.hidden_dim,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

        if self.config.use_causal_conv:
            self.causal_conv = CausalConv1d(
                features=self.config.hidden_dim,
                kernel_size=self.config.conv_kernel_size,
                param_dtype=self.config.param_dtype,
            )

        gate = partial(
            BlockDiagonalDense,
            self.config.hidden_dim,
            num_heads=self.config.num_heads,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

        self.i_gate = gate()
        self.f_gate = gate()
        self.z_gate = gate()
        self.o_gate = gate()

        recurrent_gate = partial(
            BlockDiagonalDense,
            self.config.hidden_dim,
            num_heads=self.config.num_heads,
            use_bias=False,
            kernel_init=nn.initializers.zeros_init(),
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

        self.ri = recurrent_gate()
        self.rf = recurrent_gate()
        self.rz = recurrent_gate()
        self.ro = recurrent_gate()

        self.i_bias = self.param(
            "i_bias",
            nn.initializers.zeros_init(),
            (self.config.hidden_dim,),
            self.config.param_dtype,
        )
        self.f_bias = self.param(
            "f_bias",
            powerlaw(self.config.num_heads, head_dim=head_dim),
            (self.config.hidden_dim,),
            self.config.param_dtype,
        )
        self.z_bias = self.param(
            "z_bias",
            nn.initializers.zeros_init(),
            (self.config.hidden_dim,),
            self.config.param_dtype,
        )
        self.o_bias = self.param(
            "o_bias",
            nn.initializers.zeros_init(),
            (self.config.hidden_dim,),
            self.config.param_dtype,
        )

        self.drop = nn.Dropout(rate=self.config.dropout_rate)
        self.norm = MultiHeadLayerNorm(use_scale=True, use_bias=False)
        self.output_projection = nn.Dense(
            features=self.config.features,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

    def __call__(self, carry: sLSTMCarry, inputs: Array) -> tuple[sLSTMCarry, Array]:
        c, n, m, h, buffer = carry.cell, carry.normalizer, carry.max_log, carry.hidden, carry.buffer

        B, *_ = inputs.shape
        head_dim = self.config.hidden_dim // self.config.num_heads

        x = self.input_projection(inputs)

        if self.config.use_causal_conv:
            buffer, u = self.causal_conv(add_time_axis(x), buffer)
            u = jax.nn.silu(remove_time_axis(u))
        else:
            u = x

        i = self.i_gate(u) + self.ri(h) + self.i_bias
        f = self.f_gate(u) + self.rf(h) + self.f_bias
        z = self.z_gate(x) + self.rz(h) + self.z_bias
        o = jax.nn.sigmoid(self.o_gate(x) + self.ro(h) + self.o_bias)

        log_f = -jax.nn.softplus(-f)
        log_f_plus_m = log_f + m
        m = jnp.where(jnp.all(n == 0.0, axis=-1, keepdims=True), i, jnp.maximum(log_f_plus_m, i))
        i = jnp.minimum(jnp.exp(i - m), jnp.ones_like(i))
        f = jnp.minimum(jnp.exp(log_f_plus_m - m), jnp.ones_like(f))

        c = f * c + i * nn.tanh(z)
        n = f * n + i
        h = o * (c / jnp.maximum(n, self.config.eps))

        y = self.drop(h, deterministic=not self.has_rng("dropout"))
        y = self.norm(y.reshape(B, self.config.num_heads, 1, head_dim))
        y = self.output_projection(y.reshape(B, self.config.hidden_dim))

        return sLSTMCarry(cell=c, normalizer=n, max_log=m, hidden=h, buffer=buffer), y

    @nn.nowrap
    def initialize_carry(
        self,
        key: jax.Array,
        input_shape: tuple[int, ...],
    ) -> sLSTMCarry:
        *batch_dims, _ = input_shape
        carry_init = initializers.zeros_init()

        key_c, key_n, key_h, key_m, key_buf = random.split(key, 5)
        mem_shape = (*batch_dims, self.config.hidden_dim)

        c = carry_init(key_c, mem_shape, self.config.param_dtype)
        n = carry_init(key_n, mem_shape, self.config.param_dtype)
        m = carry_init(key_m, mem_shape, self.config.param_dtype)
        h = carry_init(key_h, mem_shape, self.config.param_dtype)

        buffer = carry_init(
            key_buf, (*batch_dims, self.config.conv_kernel_size, self.config.hidden_dim)
        )

        return sLSTMCarry(cell=c, normalizer=n, max_log=m, hidden=h, buffer=buffer)

    @property
    def num_feature_axes(self) -> int:
        return 1
