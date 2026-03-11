import jax.numpy as jnp
from flax import linen as nn

from memorax.networks.initializers import bounded_uniform, kaiming_uniform


class CausalConv1d(nn.Module):
    features: int
    kernel_size: int = 4
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, state: jnp.ndarray) -> tuple:
        kernel = self.param(
            "kernel",
            kaiming_uniform(),
            (self.kernel_size, self.features),
            self.param_dtype,
        )

        conv_state = jnp.concatenate([state[:, 1:, :], x], axis=1)
        y = jnp.einsum("bkf,kf->bf", conv_state, kernel)[:, None, :]

        if self.use_bias:
            bias = self.param(
                "bias", nn.initializers.zeros_init(), (self.features,), self.param_dtype
            )
            y = y + bias
        return conv_state, y


class ParallelCausalConv1d(nn.Module):
    features: int
    kernel_size: int = 4
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        *_, feature_group_count = x.shape
        padding = self.kernel_size - 1
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            kernel_init=kaiming_uniform(),
            bias_init=bounded_uniform(
                min_val=-1.0 / jnp.sqrt(self.kernel_size),
                max_val=1.0 / jnp.sqrt(self.kernel_size),
            ),
            feature_group_count=feature_group_count,
            padding=[(padding, 0)],
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        return x
