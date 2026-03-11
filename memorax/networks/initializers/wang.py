import jax
import jax.numpy as jnp


def wang(dim, num_blocks):
    def init(key, shape, dtype):
        std = 2.0 / (num_blocks * jnp.sqrt(dim))
        return jax.random.normal(key, shape, dtype) * std

    return init
