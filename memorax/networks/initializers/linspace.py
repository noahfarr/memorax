from typing import Callable

import jax.numpy as jnp

from memorax.utils.typing import Array, Key


def linspace(start, stop) -> Callable:
    def init(key: Key, shape: tuple, dtype) -> Array:
        num_dims, *_ = shape
        return jnp.linspace(start, stop, num_dims, dtype=dtype)

    return init
