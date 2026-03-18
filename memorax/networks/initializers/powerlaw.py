from typing import Callable

import jax.numpy as jnp

from memorax.utils.typing import Array, Key


def powerlaw(num_heads, head_dim) -> Callable:
    """Initializes a weight matrix with a power law distribution."""

    def init(key: Key, shape: tuple, dtype) -> Array:
        x = jnp.arange(head_dim) / jnp.maximum(1.0, head_dim - 1)
        v = -(-5.0 + 12.0 * (x**0.3))
        b = jnp.tile(v, reps=(num_heads,))
        return b.astype(dtype)

    return init
