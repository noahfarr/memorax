import jax
import wandb

from memorax.utils.axes import ensure_axis
from memorax.utils.typing import PyTree


class WandbLogger:
    def __init__(
        self,
        entity=None,
        project=None,
        name=None,
        group=None,
        mode="disabled",
        cfg=None,
        seed=0,
        num_seeds=1,
        **kwargs,
    ):
        self.runs = {
            i: wandb.init(
                entity=entity,
                project=project,
                name=name,
                group=group,
                mode=mode,
                config={**(cfg or {}), "seed": seed + i},
                reinit="create_new",
            )
            for i in range(num_seeds)
        }

    def log(self, data: PyTree, step: int, **kwargs) -> None:
        num_seeds = len(self.runs)
        data = {
            "/".join(str(p.key) for p in path): ensure_axis(leaf, num_seeds)
            for path, leaf in jax.tree_util.tree_leaves_with_path(data)
        }
        for seed, run in self.runs.items():
            run.log({k: v[seed] for k, v in data.items()}, step=step)

    def finish(self) -> None:
        for run in self.runs.values():
            run.finish()
