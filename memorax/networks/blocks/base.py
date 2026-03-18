from typing import Protocol

from memorax.utils.typing import Array, Carry, Key


class Block(Protocol):
    """Protocol for composable neural network blocks.

    All blocks accept inputs, an optional done flag, and optional carry state,
    returning a tuple of (carry, output). Stateless blocks return None for carry.
    """

    def __call__(
        self,
        inputs: Array,
        done: Array | None = None,
        initial_carry: Carry | None = None,
        **kwargs,
    ) -> tuple[Carry, Array]: ...

    def initialize_carry(self, key: Key, input_shape: tuple) -> Carry: ...
