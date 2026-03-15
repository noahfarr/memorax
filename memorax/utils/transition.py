from typing import Optional

import jax
from flax import struct

from memorax.utils.axes import add_time_axis, remove_time_axis
from memorax.utils.timestep import Timestep
from memorax.utils.typing import PyTree


@struct.dataclass(frozen=True)
class Transition:
    first: Optional[Timestep] = None
    second: Optional[Timestep] = None
    carry: Optional[PyTree] = None
    aux: Optional[PyTree] = None

    def to_sequence(self):
        return jax.tree.map(add_time_axis, self)

    def from_sequence(self):
        return jax.tree.map(remove_time_axis, self)
