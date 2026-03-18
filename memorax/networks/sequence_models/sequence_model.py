from abc import ABC, abstractmethod
import jax
from flax import linen as nn

from memorax.utils.typing import Array, Carry, Key


class SequenceModel(ABC, nn.Module):
    @abstractmethod
    def __call__(
        self,
        inputs: Array,
        done: Array,
        initial_carry: Carry | None = None,
        **kwargs,
    ) -> tuple: ...

    @abstractmethod
    def initialize_carry(self, key: Key, input_shape: tuple) -> Carry: ...

    def local_jacobian(
        self,
        inputs: Array,
        done: Array,
        carry: Carry,
        sensitivity: dict | None = None,
        **kwargs,
    ) -> tuple[Carry, Array, dict | None]:
        """Forward pass with optional RTRL sensitivity propagation.

        Returns (next_carry, y, next_sensitivity).
        By default, calls __call__ and returns None sensitivity.
        Override to provide RTRL support.
        """
        next_carry, y = self(inputs, done, carry, **kwargs)
        return next_carry, y, None

    def initialize_sensitivity(
        self, key: jax.Array, input_shape: tuple[int, ...]
    ) -> dict | None:
        """Initialize RTRL sensitivity tensors. Returns None if unsupported."""
        return None
