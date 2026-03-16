from typing import Optional, Protocol

from memorax.utils.typing import Array, Carry


class Block(Protocol):
    """Protocol for composable neural network blocks.

    All blocks accept inputs, an optional done flag, and optional carry state,
    returning a tuple of (carry, output). Stateless blocks return None for carry.
    """

    def __call__(
        self,
        inputs: Array,
        done: Optional[Array] = None,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> tuple[Carry, Array]: ...

    def initialize_carry(self, key, input_shape) -> Carry: ...
