from typing import Callable

import jax
import jax.numpy as jnp

from memorax.utils.typing import Array, Key


def inverse_softplus(min_val: float = 0.001, max_val: float = 0.1) -> Callable:
    def init(key: Key, shape: tuple) -> Array:
        dt = jnp.exp(
            jax.random.uniform(key, shape) * (jnp.log(max_val) - jnp.log(min_val))
            + jnp.log(min_val)
        )
        return dt + jnp.log(-jnp.expm1(-dt))

    return init
