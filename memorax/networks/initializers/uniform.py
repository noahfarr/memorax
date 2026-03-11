import jax


def bounded_uniform(min_val, max_val):
    def init(key, shape, dtype):
        return jax.random.uniform(
            key, shape, minval=min_val, maxval=max_val, dtype=dtype
        )

    return init
