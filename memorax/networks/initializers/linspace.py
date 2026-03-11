import jax.numpy as jnp


def linspace(start, stop):
    def init(key, shape, dtype):
        num_dims, *_ = shape
        return jnp.linspace(start, stop, num_dims, dtype=dtype)

    return init
