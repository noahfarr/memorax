from typing import Callable

import jax
import jax.numpy as jnp

from memorax.utils.typing import Array, Key


def log_step(dt_min: float = 0.001, dt_max: float = 0.1) -> Callable:
    """Initializer for SSM discretization time steps (one per head).

    Returns an init function compatible with Flax's self.param.
    The shape argument should be (num_heads,).
    """

    def init(key, shape) -> Array:
        (h,) = shape
        keys = jax.random.split(key, h)
        return jax.vmap(
            lambda k: jax.random.uniform(k, (1,))
            * (jnp.log(dt_max) - jnp.log(dt_min))
            + jnp.log(dt_min)
        )(keys)

    return init
