import jax
import jax.numpy as jnp

from memorax.utils.typing import Array


def burn_in(network, params, timestep, carry, length):
    """Warm up carry by running network over the first `length` timesteps. No-op if length == 0."""
    if length > 0:
        timestep = jax.tree.map(lambda x: x[:, :length], timestep)
        carry, _ = network.apply(jax.lax.stop_gradient(params), *timestep, initial_carry=carry)
    return carry


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
