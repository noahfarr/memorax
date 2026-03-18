from typing import Callable

import jax
import jax.numpy as jnp

from memorax.utils.typing import Array, Key


def small(dim) -> Callable:
    def init(key: Key, shape: tuple, dtype) -> Array:
        std = jnp.sqrt(2.0 / 5.0 / dim)
        return jax.random.normal(key, shape, dtype) * std

    return init
