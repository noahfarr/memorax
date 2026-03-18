import jax.numpy as jnp

from memorax.utils.typing import Array


def broadcast(x: Array | None, to: Array) -> Array | None:
    if x is None:
        return x
    if x.ndim > to.ndim:
        raise ValueError(
            f"Cannot broadcast array with ndim={x.ndim} to target with ndim={to.ndim}. "
            "Source has more dimensions than target."
        )
    while x.ndim < to.ndim:
        x = x[..., None]
    return x
