from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from memorax.utils.typing import Array


class FeatureExtractor(nn.Module):
    observation_extractor: Callable
    action_extractor: Callable | None = None
    reward_extractor: Callable | None = None
    done_extractor: Callable | None = None

    def extract(
        self,
        embeddings: dict,
        key: str,
        extractor: Callable | None,
        x: Array | None = None,
    ) -> None:
        if extractor is not None and x is not None:
            embeddings[key] = extractor(x)

    @nn.compact
    def __call__(
        self,
        observation: Array,
        action: Array,
        reward: Array,
        done: Array,
        **kwargs,
    ) -> tuple[Array, dict]:
        embeddings = {"observation_embedding": self.observation_extractor(observation)}
        self.extract(embeddings, "action_embedding", self.action_extractor, action)
        self.extract(embeddings, "reward_embedding", self.reward_extractor, reward)
        self.extract(
            embeddings, "done_embedding", self.done_extractor, done.astype(jnp.int32)
        )
        x = jnp.concatenate([*embeddings.values()], axis=-1)

        return x, embeddings
