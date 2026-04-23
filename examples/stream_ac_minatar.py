import time
from dataclasses import asdict

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox

from memorax.algorithms import StreamAC, StreamACConfig
from memorax.environments import environment
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
    StickyActionWrapper,
)
from memorax.loggers import DashboardLogger, MultiLogger, WandbLogger
from memorax.networks import FeatureExtractor, Flatten, Network, heads
from memorax.networks.initializers import sparse

total_timesteps = 10_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "gymnax::Breakout-MinAtar"

env, env_params = environment.make(env_id)
env = StickyActionWrapper(env)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env)

num_actions = env.action_space(env_params).n

config = StreamACConfig(
    num_envs=1,
    trace_lambda=0.8,
    actor_lr=1.0,
    critic_lr=1.0,
    actor_kappa=3.0,
    critic_kappa=2.0,
    entropy_coefficient=0.01,
    gamma=0.99,
)

sparse_init = sparse(sparsity=0.9)


class LayerNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        axes = tuple(range(1, x.ndim))
        return nn.LayerNorm(
            use_bias=False,
            use_scale=False,
            epsilon=1e-5,
            reduction_axes=axes,
            use_fast_variance=False,
        )(x)


feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (
            nn.Conv(
                16, (3, 3), strides=(1, 1), padding="VALID", kernel_init=sparse_init
            ),
            LayerNorm(),
            nn.leaky_relu,
            Flatten(start_dim=-3),
            nn.Dense(128, kernel_init=sparse_init),
            LayerNorm(),
            nn.leaky_relu,
        )
    ),
)

actor_network = Network(
    feature_extractor=feature_extractor,
    head=heads.Categorical(action_dim=num_actions, kernel_init=sparse_init),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    head=heads.VNetwork(kernel_init=sparse_init),
)

agent = StreamAC(config, env, env_params, actor_network, critic_network)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={
                "Algorithm": "stream-AC",
                "Environment": env_id,
                "Total Timesteps": f"{total_timesteps:_}",
            },
        ),
        WandbLogger(
            project="memorax",
            name="stream_ac_minatar",
            mode="offline",
            cfg=None,
            seed=seed,
            num_seeds=num_seeds,
        ),
    ]
)

init = jax.vmap(agent.init)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

key = jax.random.key(seed)
key, init_key = jax.random.split(key)
state = init(jax.random.split(init_key, num_seeds))

for i in range(num_epochs):
    start = time.perf_counter()
    key, train_key = jax.random.split(key)
    state, logs = train(jax.random.split(train_key, num_seeds), state, num_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_steps / (end - start))

    info = logs.pop("info")
    mask = info["returned_episode"]
    axes = tuple(range(1, mask.ndim))
    episode_returns = jnp.mean(info["returned_episode_returns"], axis=axes, where=mask)
    episode_lengths = jnp.mean(info["returned_episode_lengths"], axis=axes, where=mask)

    data = {
        "training/SPS": SPS,
        "training/episode_returns": episode_returns,
        "training/episode_lengths": episode_lengths,
        **logs,
    }
    logger.log(data, step=state.step.mean(dtype=jnp.int32).item())

logger.finish()
