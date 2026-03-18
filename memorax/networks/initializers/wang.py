from typing import Callable

import jax
import jax.numpy as jnp

from memorax.utils.typing import Array, Key


def wang(dim, num_blocks) -> Callable:
    def init(key: Key, shape: tuple, dtype) -> Array:
        std = 2.0 / (num_blocks * jnp.sqrt(dim))
        return jax.random.normal(key, shape, dtype) * std

    return init
