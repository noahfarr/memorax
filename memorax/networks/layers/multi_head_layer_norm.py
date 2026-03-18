import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype

from memorax.utils.typing import Array


class MultiHeadLayerNorm(nn.Module):
    eps: float = 1e-5
    use_scale: bool = True
    use_bias: bool = False
    residual_weight: bool = True
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x) -> Array:
        B, NH, S, DH = x.shape

        y = nn.vmap(
            nn.LayerNorm,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=1,
            out_axes=1,
        )(
            epsilon=self.eps,
            use_scale=False,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(
            x
        )

        if self.use_scale:
            gamma = self.param(
                "weight", nn.initializers.zeros_init(), (NH, DH), self.param_dtype
            )
            scale = (1.0 + gamma) if self.residual_weight else gamma
            y = y * scale[None, :, None, :].astype(y.dtype)

        if self.use_bias:
            beta = self.param(
                "bias", nn.initializers.zeros_init(), (NH, DH), self.param_dtype
            )
            y = y + beta[None, :, None, :].astype(y.dtype)

        return y
