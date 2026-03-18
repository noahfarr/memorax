from typing import Callable

import jax

from memorax.utils.typing import Array, Key


def bounded_uniform(min_val, max_val) -> Callable:
    def init(key: Key, shape: tuple, dtype) -> Array:
        return jax.random.uniform(
            key, shape, minval=min_val, maxval=max_val, dtype=dtype
        )

    return init
