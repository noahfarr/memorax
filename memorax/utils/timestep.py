from typing import Self

import jax
from flax import struct

from memorax.utils.axes import add_feature_axis, add_time_axis, remove_time_axis
from memorax.utils.typing import Array


@struct.dataclass(frozen=True)
class Timestep:
    obs: Array | None = None
    action: Array | None = None
    reward: Array | None = None
    done: Array | None = None

    def __iter__(self):
        yield self.obs
        yield self.done
        yield self.action
        yield add_feature_axis(self.reward)

    def to_sequence(self) -> Self:
        return jax.tree.map(add_time_axis, self)

    def from_sequence(self) -> Self:
        return jax.tree.map(remove_time_axis, self)
