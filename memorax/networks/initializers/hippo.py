from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.nn.initializers import lecun_normal


def truncated_standard_normal(key, shape):
    h, p, _ = shape
    cs = []
    for _ in range(h):
        key, sk = jax.random.split(key)
        cs.append(lecun_normal()(sk, (1, p, 2)))
    return jnp.asarray(cs)[:, 0]


def init_v_inv_b(
    init_fun: Callable, rng: jax.Array, shape: Tuple[int, int], vinv: jnp.ndarray
) -> jnp.ndarray:
    b = init_fun(rng, shape)
    vinv_b = vinv.astype(jnp.complex64) @ b.astype(jnp.complex64)
    r = vinv_b.real
    i = vinv_b.imag
    return jnp.concatenate([r[..., None], i[..., None]], axis=-1)


def init_cv(
    init_fun: Callable, rng: jax.Array, shape: Tuple[int, int, int], v: jnp.ndarray
) -> jnp.ndarray:
    c_ = init_fun(rng, shape)
    c = c_[..., 0] + 1j * c_[..., 1]
    cv = c @ v.astype(jnp.complex64)
    r = cv.real
    i = cv.imag
    return jnp.concatenate([r[..., None], i[..., None]], axis=-1)
