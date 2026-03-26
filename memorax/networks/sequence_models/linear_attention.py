from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.typing import Dtype, Initializer

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


@struct.dataclass
class LinearAttentionConfig:
    features: int
    head_dim: int
    num_heads: int
    kernel_init: Initializer = struct.field(
        pytree_node=False, default=nn.initializers.lecun_normal()
    )
    bias_init: Initializer = struct.field(
        pytree_node=False, default=nn.initializers.zeros_init()
    )
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    eps: float = 1e-6


@struct.dataclass
class LinearAttentionCarry:
    S: Array
    z: Array


class LinearAttentionCell(MemoroidCellBase):
    config: LinearAttentionConfig

    def setup(self):
        projection = partial(
            nn.DenseGeneral,
            features=(self.config.num_heads, self.config.head_dim),
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )

        self.key = projection(name="key")
        self.value = projection(name="value")
        self.query = projection(name="query")
        self.norm = nn.RMSNorm(dtype=self.config.dtype)
        self.output_projection = nn.Dense(
            features=self.config.features,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )

    def _feature_map(self, x: Array) -> Array:
        return nn.elu(x) + 1

    def __call__(self, x: Array, **kwargs) -> Carry:
        key = self._feature_map(self.key(x))
        value = self.value(x)

        S = jnp.einsum("bthi,bthj->bthij", value, key)

        return LinearAttentionCarry(S=S, z=key)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        return LinearAttentionCarry(
            S=a.S + b.S,
            z=a.z + b.z,
        )

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        B, T, _ = x.shape

        query = self._feature_map(self.query(x))

        numerator = jnp.einsum("bthij,bthj->bthi", h.S, query)

        denominator = jnp.einsum("bthi,bthi->bth", query, h.z)
        denominator = jnp.maximum(denominator, self.config.eps)[:, :, :, None]

        output = numerator / denominator

        hidden_dim = self.config.num_heads * self.config.head_dim
        output = output.reshape(B, T, hidden_dim)

        output = self.norm(output)
        output = self.output_projection(output)

        return output

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        S = jnp.zeros(
            (
                *batch_dims,
                1,
                self.config.num_heads,
                self.config.head_dim,
                self.config.head_dim,
            )
        )
        z = jnp.zeros((*batch_dims, 1, self.config.num_heads, self.config.head_dim))
        return LinearAttentionCarry(S=S, z=z)
