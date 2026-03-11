import jax.numpy as jnp


def powerlaw(num_heads, head_dim):
    """Initializes a weight matrix with a power law distribution."""

    def init(key, shape, dtype):
        x = jnp.arange(head_dim) / jnp.maximum(1.0, head_dim - 1)
        v = -(-5.0 + 12.0 * (x**0.3))
        b = jnp.tile(v, reps=(num_heads,))
        return b.astype(dtype)

    return init
