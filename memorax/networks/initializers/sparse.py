import math

import jax
import jax.numpy as jnp


def sparse(sparsity: float = 0.9):
    """LeCun uniform initialization with sparse connectivity.

    For each output unit, zeros out `sparsity` fraction of input connections
    chosen at random. Remaining weights are drawn from U(-1/√fan_in, 1/√fan_in).

    Args:
        sparsity: Fraction of weights to zero out per output unit (default: 0.9).
    """

    def init(key, shape, dtype=jnp.float32):
        fan_in = math.prod(shape[:-1])
        fan_out = shape[-1]
        limit = math.sqrt(1.0 / fan_in)

        key, weight_key = jax.random.split(key)
        weights = jax.random.uniform(
            weight_key, shape, dtype, minval=-limit, maxval=limit
        )

        n_zero = int(sparsity * fan_in)
        weights_flat = weights.reshape(fan_in, fan_out)

        perms = jax.vmap(lambda k: jax.random.permutation(k, fan_in))(
            jax.random.split(key, fan_out)
        )
        mask = (perms >= n_zero).astype(dtype).T  # (fan_in, fan_out)

        return (weights_flat * mask).reshape(shape)

    return init
