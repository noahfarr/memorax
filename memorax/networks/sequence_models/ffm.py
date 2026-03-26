from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen import initializers
from flax.linen.linear import default_kernel_init
from flax.typing import Dtype, Initializer

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


@struct.dataclass
class FFMConfig:
    features: int
    memory_size: int
    context_size: int
    min_period: int = 1
    max_period: int = 1024
    epsilon: float = 0.01
    beta: float = 0.01
    kernel_init: Initializer = struct.field(pytree_node=False, default=default_kernel_init)
    bias_init: Initializer = struct.field(pytree_node=False, default=initializers.zeros_init())
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32


@struct.dataclass
class FFMCarry:
    memory: Array
    time: Array


class FFMCell(MemoroidCellBase):
    config: FFMConfig

    def setup(self):
        self.limit = (
            jnp.log(jnp.finfo(self.config.param_dtype).max) / self.config.max_period - self.config.epsilon
        )

        low = -self.limit + self.config.epsilon
        high = jnp.maximum(
            jnp.minimum(-1e-6, jnp.log(self.config.beta) / self.config.max_period), low
        )
        self.alpha = self.param(
            "alpha",
            lambda _: jnp.linspace(low, high, self.config.memory_size, dtype=self.config.param_dtype),
        )
        self.omega = self.param(
            "omega",
            lambda _: (2 * jnp.pi)
            / jnp.linspace(
                self.config.min_period,
                self.config.max_period,
                self.config.context_size,
                dtype=self.config.param_dtype,
            ),
        )

        dense = partial(
            nn.Dense,
            use_bias=True,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

        self.pre = dense(features=self.config.memory_size, name="pre")
        self.input_gate = dense(features=self.config.memory_size, name="input_gate")
        self.output_gate = dense(features=self.config.features, name="output_gate")
        self.skip = dense(features=self.config.features, name="skip")
        self.mix = dense(features=self.config.features, name="mix")
        self.ln = nn.LayerNorm(use_scale=False, use_bias=False, name="ln")

    def _complex_dtype(self):
        return jnp.complex64 if self.config.param_dtype == jnp.float32 else jnp.complex128

    def _gamma(self, dt):
        alpha = jnp.clip(self.alpha, min=-self.limit, max=-1e-8)
        alpha = alpha.reshape(1, self.config.memory_size, 1)
        omega = self.omega.reshape(1, 1, self.config.context_size)
        return jnp.exp(jax.lax.complex(alpha, omega) * dt[..., None, None])

    def __call__(self, x: Array, **kwargs) -> Carry:
        B, T, _ = x.shape

        pre = self.pre(x)
        gate = nn.sigmoid(self.input_gate(x))
        x_tilde = pre * gate

        memory = jnp.repeat(x_tilde[..., None], self.config.context_size, axis=-1)
        memory = jax.lax.complex(memory, jnp.zeros_like(memory))

        time = jnp.arange(T, dtype=self.config.param_dtype)
        time = jnp.broadcast_to(time, (B, T))
        time = jax.lax.complex(time, jnp.zeros_like(time))

        return FFMCarry(memory=memory, time=time)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        dt = b.time - a.time
        gamma = self._gamma(dt)
        return FFMCarry(memory=a.memory * gamma + b.memory, time=b.time)

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        output_gate = nn.sigmoid(self.output_gate(x))
        skip = self.skip(x)

        z = jnp.concatenate([jnp.real(h.memory), jnp.imag(h.memory)], axis=-1)
        z = z.reshape((*z.shape[:-2], -1))
        z = self.mix(z)

        y = self.ln(z) * output_gate + skip * (1.0 - output_gate)
        return y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        memory = jnp.zeros(
            (*batch_dims, 1, self.config.memory_size, self.config.context_size),
            dtype=self._complex_dtype(),
        )
        time = jnp.full((*batch_dims, 1), -1, dtype=self._complex_dtype())
        return FFMCarry(memory=memory, time=time)
