from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.typing import Dtype, Initializer

from memorax.networks.initializers import inverse_softplus, log_uniform
from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


@struct.dataclass
class Mamba2Config:
    features: int
    num_heads: int = 8
    head_dim: int = 16
    state_dim: int = 16
    num_groups: int = 1
    conv_dim: int = 4
    kernel_init: Initializer = struct.field(
        pytree_node=False, default=nn.initializers.lecun_normal()
    )
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32


@struct.dataclass
class Mamba2Carry:
    state: Array
    decay: Array


class Mamba2Cell(MemoroidCellBase):
    config: Mamba2Config

    def setup(self):
        assert self.config.num_heads % self.config.num_groups == 0
        hidden_dim = self.config.num_heads * self.config.head_dim
        group_projection_dim = self.config.num_groups * self.config.state_dim
        conv_channels = hidden_dim + 2 * group_projection_dim

        self.A_log = self.param("A_log", log_uniform(), (self.config.num_heads,))
        self.D = self.param("D", nn.initializers.ones, (self.config.num_heads,))
        self.dt_bias = self.param("dt_bias", inverse_softplus(), (self.config.num_heads,))

        projection = partial(
            nn.Dense,
            kernel_init=self.config.kernel_init,
            use_bias=False,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

        self.input_projection = projection(hidden_dim * 2)
        self.B = projection(group_projection_dim)
        self.C = projection(group_projection_dim)
        self.dt = projection(self.config.num_heads)
        self.conv = nn.Conv(
            conv_channels,
            kernel_size=(self.config.conv_dim,),
            padding=((self.config.conv_dim - 1, 0),),
            feature_group_count=conv_channels,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )
        self.norm = nn.RMSNorm(self.config.num_heads * self.config.head_dim)
        self.output_projection = nn.Dense(
            self.config.features,
            kernel_init=self.config.kernel_init,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
        )

    def _project(self, x: Array):
        batch_size, sequence_length, _ = x.shape
        hidden_dim = self.config.num_heads * self.config.head_dim
        group_projection_dim = self.config.num_groups * self.config.state_dim
        heads_per_group = self.config.num_heads // self.config.num_groups

        hidden, gate = jnp.split(self.input_projection(x), 2, axis=-1)

        B = self.B(x)
        C = self.C(x)

        conv_input = jnp.concatenate([hidden, B, C], axis=-1)
        conv_input = nn.silu(self.conv(conv_input))

        hidden = conv_input[..., :hidden_dim].reshape(
            batch_size,
            sequence_length,
            self.config.num_groups,
            heads_per_group,
            self.config.head_dim,
        )
        B = conv_input[..., hidden_dim : hidden_dim + group_projection_dim].reshape(
            batch_size, sequence_length, self.config.num_groups, self.config.state_dim
        )
        C = conv_input[..., hidden_dim + group_projection_dim :].reshape(
            batch_size, sequence_length, self.config.num_groups, self.config.state_dim
        )

        dt = nn.softplus(self.dt(x) + self.dt_bias).reshape(
            batch_size, sequence_length, self.config.num_groups, heads_per_group
        )

        return hidden, B, C, gate, dt

    def __call__(self, x: Array, **kwargs) -> Carry:
        hidden, B, _, _, dt = self._project(x)

        A = -jnp.exp(self.A_log).reshape(
            self.config.num_groups, self.config.num_heads // self.config.num_groups
        )
        decay = jnp.exp(dt * A)[..., None, None]

        h = jnp.einsum("btgn,btgp,btgpd->btgpnd", B, dt, hidden)

        return Mamba2Carry(state=h, decay=decay)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        return Mamba2Carry(
            state=b.decay * a.state + b.state,
            decay=b.decay * a.decay,
        )

    def read(self, carry: Carry, x: Array, **kwargs) -> Array:
        batch_size, sequence_length, _ = x.shape

        hidden, _, C, gate, _ = self._project(x)

        y = jnp.einsum("btgn,btgpnd->btgpd", C, carry.state)
        D = self.D.reshape(
            self.config.num_groups, self.config.num_heads // self.config.num_groups
        )
        y = y + jnp.einsum("gp,btgpd->btgpd", D, hidden)
        y = y.reshape(
            batch_size, sequence_length, self.config.num_heads * self.config.head_dim
        )
        y = self.norm(y) * nn.silu(gate)

        return self.output_projection(y)

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        heads_per_group = self.config.num_heads // self.config.num_groups
        state = jnp.zeros(
            (
                *batch_dims,
                1,
                self.config.num_groups,
                heads_per_group,
                self.config.state_dim,
                self.config.head_dim,
            ),
            dtype=self.config.dtype,
        )
        decay = jnp.ones(
            (*batch_dims, 1, self.config.num_groups, heads_per_group, 1, 1),
            dtype=self.config.dtype,
        )
        return Mamba2Carry(state=state, decay=decay)
