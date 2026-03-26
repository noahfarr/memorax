from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen import initializers
from flax.linen.recurrent import RNNCellBase
from flax.typing import Dtype, Initializer
from jax import random

from memorax.utils.typing import Array

from memorax.networks.initializers import xavier_uniform


@struct.dataclass
class SHMConfig:
    features: int
    output_features: int
    num_thetas: int = 128
    sample_theta: bool = True
    kernel_init: Initializer = struct.field(pytree_node=False, default=nn.initializers.lecun_normal())
    bias_init: Initializer = struct.field(pytree_node=False, default=initializers.zeros_init())
    theta_init: Initializer = struct.field(pytree_node=False, default=xavier_uniform())
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = struct.field(pytree_node=False, default=initializers.zeros_init())


@struct.dataclass
class SHMCarry:
    memory: Array


class SHMCell(RNNCellBase):
    config: SHMConfig

    def setup(self):
        dense = partial(
            nn.Dense,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )

        self.ln = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )
        self.value = dense(features=self.config.features)
        self.key = dense(features=self.config.features)
        self.query = dense(features=self.config.features)
        self.v_c = dense(features=self.config.features)
        self.eta = dense(features=1)
        self.theta_table = self.param(
            "theta_table",
            self.config.theta_init,
            (self.config.num_thetas, self.config.features),
            self.config.param_dtype,
        )
        self.output_projection = nn.Dense(self.config.output_features)

    def __call__(self, carry: SHMCarry, inputs: Array) -> tuple[SHMCarry, Array]:
        inputs = self.ln(inputs)

        value = self.value(inputs)
        key = jax.nn.relu(self.key(inputs))
        query = jax.nn.relu(self.query(inputs))
        v_c = self.v_c(inputs)
        eta_val = nn.sigmoid(self.eta(inputs))

        key = key / (1e-5 + jnp.sum(key, axis=-1, keepdims=True))
        query = query / (1e-5 + jnp.sum(query, axis=-1, keepdims=True))

        U = jnp.einsum('...i,...j->...ij', eta_val * value, key)

        if self.config.sample_theta and self.has_rng("torso"):
            rng = self.make_rng("torso")
            batch_shape = v_c.shape[:-1]
            idx = random.randint(rng, batch_shape, 0, self.config.num_thetas, dtype=jnp.int32)
            theta_t = self.theta_table[idx]
            theta_t = jnp.broadcast_to(theta_t, v_c.shape)
        else:
            theta_t = jnp.broadcast_to(self.theta_table[0], v_c.shape)

        C = 1.0 + jnp.tanh(jnp.einsum('...i,...j->...ij', theta_t, v_c))

        M = carry.memory * C + U

        h = jnp.einsum("...ij,...j->...i", M, query)

        y = self.output_projection(h)

        return SHMCarry(memory=M), y

    @nn.nowrap
    def initialize_carry(self, key: jax.Array, input_shape: tuple[int, ...]) -> SHMCarry:
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.config.features, self.config.features)
        return SHMCarry(memory=self.config.carry_init(key, mem_shape, self.config.param_dtype))

    @property
    def num_feature_axes(self) -> int:
        return 1
