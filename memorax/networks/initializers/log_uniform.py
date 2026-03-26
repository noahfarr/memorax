from typing import Callable

import jax
import jax.numpy as jnp

from memorax.utils.typing import Array, Key


def log_uniform(min_val: float = 1.0, max_val: float = 16.0) -> Callable:
    def init(key: Key, shape: tuple) -> Array:
        return jax.random.uniform(
            key, shape, minval=jnp.log(min_val), maxval=jnp.log(max_val)
        )

    return init
