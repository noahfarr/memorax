from typing import Callable

import flax.linen as nn

from memorax.utils.typing import Array, Carry, Key

from .base import Block


class Residual(nn.Module, Block):
    """Wraps a module with a residual connection: output = x + module(x)."""

    module: nn.Module

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        done: Array | None = None,
        initial_carry: Carry | None = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        carry, output = self.module(
            inputs, done=done, initial_carry=initial_carry, **kwargs
        )
        return carry, inputs + output

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple) -> Carry:
        return self.module.initialize_carry(key, input_shape)


class GatedResidual(nn.Module, Block):
    """Residual connection with a learned gate: output = x + gate * module(x)."""

    module: nn.Module
    gate: Callable = nn.sigmoid

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        done: Array | None = None,
        initial_carry: Carry | None = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        features = inputs.shape[-1]

        carry, output = self.module(
            inputs, done=done, initial_carry=initial_carry, **kwargs
        )

        gate = nn.Dense(
            features,
            name="gate",
        )(inputs)
        gate = self.gate(gate)

        return carry, inputs + gate * output

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple) -> Carry:
        return self.module.initialize_carry(key, input_shape)
