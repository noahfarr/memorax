import jax
import numpy as np
import orbax.checkpoint.experimental.v1 as ocp

from memorax.utils.decorators import callback
from memorax.utils.typing import PyTree


class CheckpointLogger:
    def __init__(self, directory="checkpoints", max_to_keep=None, **kwargs):
        preservation_policy = None
        if max_to_keep is not None:
            preservation_policy = ocp.training.preservation_policies.LatestN(
                max_to_keep
            )
        self.checkpointer = ocp.training.Checkpointer(
            directory,
            preservation_policy=preservation_policy,
        )

    @callback
    def log(self, data: PyTree, step: int, train_state: PyTree | None = None, **kwargs):
        if train_state is None:
            return
        train_state = jax.tree.map(lambda value: np.asarray(value), train_state)
        self.checkpointer.save_pytree(int(step), train_state, force=True)

    def finish(self) -> None:
        self.checkpointer.close()
