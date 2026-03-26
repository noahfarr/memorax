from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen import initializers
from flax.typing import Dtype, Initializer

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


@struct.dataclass
class MinGRUConfig:
    features: int
    kernel_init: Initializer = struct.field(
        pytree_node=False, default=initializers.lecun_normal()
    )
    bias_init: Initializer = struct.field(
        pytree_node=False, default=initializers.zeros_init()
    )
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32


@struct.dataclass
class MinGRUCarry:
    log_state: Array
    decay: Array


class MinGRUCell(MemoroidCellBase):
    config: MinGRUConfig

    def setup(self):
        self.z = nn.Dense(
            features=self.config.features,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            name="z",
        )
        self.h = nn.Dense(
            features=self.config.features,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            name="h",
        )

    def __call__(self, x: Array, **kwargs) -> Carry:
        B, T, _ = x.shape

        z = self.z(x)
        h_tilde = self.h(x)

        log_z = -nn.softplus(-z)
        log_h_tilde = jnp.where(
            h_tilde >= 0, jnp.log(nn.relu(h_tilde) + 0.5), -nn.softplus(-h_tilde)
        )

        log_state = log_z + log_h_tilde
        decay = -nn.softplus(z)

        return MinGRUCarry(log_state=log_state, decay=decay)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        return MinGRUCarry(
            log_state=jnp.logaddexp(b.decay + a.log_state, b.log_state),
            decay=a.decay + b.decay,
        )

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        return jnp.exp(h.log_state)

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        log_state = jnp.full(
            (*batch_dims, 1, self.config.features),
            -jnp.inf,
            dtype=self.config.dtype or self.config.param_dtype,
        )
        decay = jnp.zeros(
            (*batch_dims, 1, self.config.features),
            dtype=self.config.dtype or self.config.param_dtype,
        )
        return MinGRUCarry(log_state=log_state, decay=decay)
