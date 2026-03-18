import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, Initializer

from memorax.networks.initializers import small
from memorax.utils.typing import Array


class BlockDiagonalDense(nn.Module):
    features: int
    num_heads: int
    use_bias: bool = True
    kernel_init: Initializer | None = None
    bias_init: Initializer = nn.initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        *batch, features = x.shape
        block_size = features // self.num_heads

        kernel_init = self.kernel_init or small(block_size)
        kernel = self.param(
            "kernel",
            kernel_init,
            (self.num_heads, block_size, block_size),
            self.param_dtype,
        )
        x = x.reshape(*batch, self.num_heads, -1)
        x = jnp.einsum("...hd,hod->...ho", x, kernel)
        x = x.reshape(*batch, -1)

        if self.use_bias:
            bias = self.param(
                "bias",
                self.bias_init,
                (self.features,),
                self.param_dtype,
            )
            bias = jnp.broadcast_to(bias, x.shape)
            x = x + bias

        return x
