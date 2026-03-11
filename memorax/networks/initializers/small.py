import jax
import jax.numpy as jnp


def small(dim):
    def init(key, shape, dtype):
        std = jnp.sqrt(2.0 / 5.0 / dim)
        return jax.random.normal(key, shape, dtype) * std

    return init
