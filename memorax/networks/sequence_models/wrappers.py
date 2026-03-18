import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from memorax.utils.typing import Array, Carry, Key

from .sequence_model import SequenceModel


class SequenceModelWrapper(SequenceModel, nn.Module):
    network: nn.Module

    def __call__(self, inputs: Array, done: Array, initial_carry: Carry | None = None, **kwargs) -> tuple[Carry, Array]:
        carry = initial_carry
        return carry, self.network(inputs, **kwargs)

    def initialize_carry(self, key: Key, input_shape: tuple) -> None:
        return None


@struct.dataclass
class RL2State:
    carry: Array
    step: Array


class RL2Wrapper(SequenceModel, nn.Module):
    sequence_model: nn.Module
    steps_per_trial: int

    def __call__(self, inputs: Array, done: Array, initial_carry: Carry | None = None, **kwargs) -> tuple[RL2State, Array]:
        _, sequence_length, *_ = inputs.shape

        if initial_carry is None:
            initial_carry = self.initialize_carry(jax.random.key(0), inputs.shape)

        time_indices = jnp.arange(sequence_length)

        steps = initial_carry.step[:, None] + time_indices[None, :]

        done = steps % self.steps_per_trial == 0

        carry, outputs = self.sequence_model(inputs, done, initial_carry.carry)
        carry = RL2State(carry=carry, step=initial_carry.step + sequence_length)
        return carry, outputs

    def initialize_carry(self, key: Key, input_shape: tuple) -> RL2State:
        batch_size, *_, features = input_shape
        return RL2State(
            carry=self.sequence_model.initialize_carry(key, (batch_size, features)),
            step=jnp.zeros((batch_size,), dtype=jnp.int32),
        )
