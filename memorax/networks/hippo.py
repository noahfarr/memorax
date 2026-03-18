import jax.numpy as jnp

from memorax.utils.typing import Array


def make_hippo(n: int) -> Array:
    p = jnp.sqrt(1.0 + 2.0 * jnp.arange(n))
    a = p[:, None] * p[None, :]
    a = jnp.tril(a) - jnp.diag(jnp.arange(n))
    return -a


def make_nplr_hippo(n: int) -> tuple[Array, Array, Array]:
    a = make_hippo(n)
    p = jnp.sqrt(jnp.arange(n) + 0.5)
    b = jnp.sqrt(2.0 * jnp.arange(n) + 1.0)
    return a, p, b


def make_dplr_hippo(
    n: int,
) -> tuple[Array, Array, Array, Array, Array]:
    a, p, b = make_nplr_hippo(n)
    s = a + p[:, None] * p[None, :]
    s_diag = jnp.diag(s)
    lambda_real = jnp.mean(s_diag) * jnp.ones_like(s_diag)
    lambda_imag, v = jnp.linalg.eigh(s * (-1j))
    p = v.conj().T @ p
    b_orig = b
    b = v.conj().T @ b
    return lambda_real + 1j * lambda_imag, p, b, v, b_orig


def discretize_bilinear(
    lam: Array, b_tilde: Array, delta: Array
) -> tuple[Array, Array]:
    ident = jnp.ones(lam.shape[0], dtype=jnp.complex64)
    bl = 1.0 / (ident - (delta / 2.0) * lam)
    lambda_bar = bl * (ident + (delta / 2.0) * lam)
    b_bar = (bl * delta)[..., None] * b_tilde
    return lambda_bar, b_bar


def discretize_zoh(
    lam: Array, b_tilde: Array, delta: Array
) -> tuple[Array, Array]:
    ident = jnp.ones(lam.shape[0], dtype=jnp.complex64)
    lambda_bar = jnp.exp(lam * delta)
    b_bar = (1.0 / lam * (lambda_bar - ident))[..., None] * b_tilde
    return lambda_bar, b_bar
