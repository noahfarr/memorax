from typing import Callable

import flax.linen as nn

from memorax.utils.typing import Array, Carry, Key

from .base import Block


class PreNorm(nn.Module, Block):
    """Applies normalization before the module: output = module(norm(x)).

    Args:
        module: The module to wrap.
        norm: Normalization class (default: nn.LayerNorm).
        norm_kwargs: Additional kwargs passed to the norm constructor.
    """

    module: nn.Module
    norm: Callable = nn.LayerNorm

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        done: Array | None = None,
        initial_carry: Carry | None = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        x = self.norm()(inputs)
        return self.module(x, done=done, initial_carry=initial_carry, **kwargs)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple) -> Carry:
        return self.module.initialize_carry(key, input_shape)


class PostNorm(nn.Module, Block):
    """Applies normalization after the module: output = norm(module(x)).

    Args:
        module: The module to wrap.
        norm: Normalization class (default: nn.LayerNorm).
        norm_kwargs: Additional kwargs passed to the norm constructor.
    """

    module: nn.Module
    norm: Callable = nn.LayerNorm

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
        return carry, self.norm()(output)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple) -> Carry:
        return self.module.initialize_carry(key, input_shape)
