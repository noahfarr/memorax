from typing import Self

import jax
from flax import struct

from memorax.utils.axes import add_time_axis, remove_time_axis
from memorax.utils.timestep import Timestep
from memorax.utils.typing import PyTree


@struct.dataclass(frozen=True)
class Transition:
    first: Timestep | None = None
    second: Timestep | None = None
    carry: PyTree | None = None
    aux: PyTree | None = None

    def to_sequence(self) -> Self:
        return jax.tree.map(add_time_axis, self)

    def from_sequence(self) -> Self:
        return jax.tree.map(remove_time_axis, self)
