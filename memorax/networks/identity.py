import flax.linen as nn

from memorax.utils.typing import Array


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x, *args, **kwargs) -> Array:
        return x
