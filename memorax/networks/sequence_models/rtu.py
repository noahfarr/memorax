from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.typing import Dtype
from jax.nn.initializers import lecun_normal

from memorax.utils.typing import Array, Carry

from .rnn import RNNCellBase


def _initialize_nu_log(key, shape, r_min=0.0, r_max=1.0):
    u = jax.random.uniform(key, shape=shape)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def _initialize_theta_log(key, shape, max_phase=6.28):
    u = jax.random.uniform(key, shape=shape)
    return jnp.log(max_phase * u)


@struct.dataclass
class RTUConfig:
    features: int
    hidden_dim: int
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    eps: float = 1e-8
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    activation_fn: Callable = struct.field(pytree_node=False, default=jnp.tanh)


@struct.dataclass
class RTUCarry:
    real: Array
    imaginary: Array


class RTUCell(RNNCellBase):

    config: RTUConfig

    @property
    def num_feature_axes(self) -> int:
        return 1

    def setup(self):
        self.nu_log = self.param(
            "nu_log",
            partial(_initialize_nu_log, r_min=self.config.r_min, r_max=self.config.r_max),
            (self.config.hidden_dim,),
        )
        self.theta_log = self.param(
            "theta_log",
            partial(_initialize_theta_log, max_phase=self.config.max_phase),
            (self.config.hidden_dim,),
        )
        self.B_real = self.param(
            "B_real",
            lecun_normal(),
            (self.config.hidden_dim, self.config.features),
        )
        self.B_imag = self.param(
            "B_imag",
            lecun_normal(),
            (self.config.hidden_dim, self.config.features),
        )

    def _g_phi_norm(self) -> tuple[Array, Array, Array, Array]:
        r = jnp.exp(-jnp.exp(self.nu_log))
        theta = jnp.exp(self.theta_log)
        g = r * jnp.cos(theta)
        phi = r * jnp.sin(theta)
        norm = jnp.sqrt(1 - r**2) + self.config.eps
        return g, phi, norm, r

    @nn.compact
    def __call__(self, carry: RTUCarry, inputs: Array) -> tuple[RTUCarry, Array]:
        g, phi, norm, _ = self._g_phi_norm()

        pre_real = g * carry.real - phi * carry.imaginary + norm * (inputs @ self.B_real.T)
        pre_imaginary = g * carry.imaginary + phi * carry.real + norm * (inputs @ self.B_imag.T)
        f = self.config.activation_fn
        new_carry = RTUCarry(real=f(pre_real), imaginary=f(pre_imaginary))
        output = jnp.concatenate([new_carry.real, new_carry.imaginary], axis=-1)
        return new_carry, output

    @nn.nowrap
    def initialize_carry(self, key: jax.Array, input_shape: tuple[int, ...]) -> RTUCarry:
        *batch_dims, _ = input_shape
        zeros = jnp.zeros((*batch_dims, self.config.hidden_dim))
        return RTUCarry(real=zeros, imaginary=zeros)

    def compute_phantom(self, sensitivity: dict[str, Array]) -> RTUCarry:
        params = self.variables["params"]
        real_phantom = 0
        imaginary_phantom = 0
        for name, S in sensitivity.items():
            param = params[name]
            diff = param - jax.lax.stop_gradient(param)
            contribution = jnp.sum(S * diff, axis=tuple(range(3, S.ndim)))
            real_phantom = real_phantom + contribution[:, 0]
            imaginary_phantom = imaginary_phantom + contribution[:, 1]
        return RTUCarry(real=real_phantom, imaginary=imaginary_phantom)

    def inject_phantom(self, carry: RTUCarry, phantom: RTUCarry) -> RTUCarry:
        return RTUCarry(
            real=jax.lax.stop_gradient(carry.real) + phantom.real,
            imaginary=jax.lax.stop_gradient(carry.imaginary) + phantom.imaginary,
        )

    def local_jacobian(
        self,
        carry: RTUCarry,
        inputs: Array,
        sensitivity: dict[str, Array],
        **kwargs,
    ) -> tuple[RTUCarry, Array, dict[str, Array]]:
        g, phi, norm, r = self._g_phi_norm()

        f = self.config.activation_fn

        u_real = inputs @ self.B_real.T
        u_imaginary = inputs @ self.B_imag.T
        pre_real = g * carry.real - phi * carry.imaginary + norm * u_real
        pre_imaginary = g * carry.imaginary + phi * carry.real + norm * u_imaginary
        d_real = jax.grad(lambda x: f(x).sum())(pre_real)
        d_imaginary = jax.grad(lambda x: f(x).sum())(pre_imaginary)
        new_carry = RTUCarry(real=f(pre_real), imaginary=f(pre_imaginary))
        output = jnp.concatenate([new_carry.real, new_carry.imaginary], axis=-1)

        A = jnp.stack([jnp.stack([g, -phi]), jnp.stack([phi, g])])
        d = jnp.stack([d_real, d_imaginary], axis=1)

        exp_nu = jnp.exp(self.nu_log)
        dg_dnu = -exp_nu * g
        dphi_dnu = -exp_nu * phi
        dnorm_dnu = exp_nu * r**2 / (jnp.sqrt(1 - r**2) + 1e-12)

        theta = jnp.exp(self.theta_log)
        dg_dtheta = -phi * theta
        dphi_dtheta = g * theta

        Bu = jnp.einsum('h,bf->bhf', norm, inputs)
        zeros_bhf = jnp.zeros_like(Bu)
        jacobians = {
            "nu_log": jnp.stack([
                dg_dnu * carry.real - dphi_dnu * carry.imaginary + dnorm_dnu * u_real,
                dg_dnu * carry.imaginary + dphi_dnu * carry.real + dnorm_dnu * u_imaginary,
            ], axis=1),
            "theta_log": jnp.stack([
                dg_dtheta * carry.real - dphi_dtheta * carry.imaginary,
                dg_dtheta * carry.imaginary + dphi_dtheta * carry.real,
            ], axis=1),
            "B_real": jnp.stack([Bu, zeros_bhf], axis=1),
            "B_imag": jnp.stack([zeros_bhf, Bu], axis=1),
        }

        next_sensitivity = {}
        for name in sensitivity:
            S = sensitivity[name]
            J = jacobians[name]
            rotated = jnp.einsum('ijh,bjh...->bih...', A, S)
            next_sensitivity[name] = jnp.einsum('bih,bih...->bih...', d, rotated + J)

        return new_carry, output, next_sensitivity

    def initialize_sensitivity(
        self, key: jax.Array, input_shape: tuple[int, ...]
    ) -> dict[str, Array] | None:
        *batch_dims, _ = input_shape
        H = self.config.hidden_dim
        F = self.config.features
        return {
            "nu_log": jnp.zeros((*batch_dims, 2, H)),
            "theta_log": jnp.zeros((*batch_dims, 2, H)),
            "B_real": jnp.zeros((*batch_dims, 2, H, F)),
            "B_imag": jnp.zeros((*batch_dims, 2, H, F)),
        }
