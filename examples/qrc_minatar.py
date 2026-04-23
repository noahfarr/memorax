import time
from dataclasses import asdict

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax

from memorax.algorithms import QRC, QRCConfig
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

total_timesteps = 5_000_000
num_epochs = 50
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 5
num_envs = 1
env_id = "gymnax::Asterix-MinAtar"

env, env_params = environment.make(env_id)
env = StickyActionWrapper(env)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env)

num_actions = env.action_space(env_params).n

cfg = QRCConfig(
    num_envs=num_envs,
    gamma=0.99,
    lamda=0.8,
    gradient_correction=True,
    reg_coeff=1.0,
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


q_network = Network(
    feature_extractor=feature_extractor,
    head=heads.DiscreteQNetwork(
        action_dim=num_actions,
        kernel_init=sparse_init,
    ),
)
h_network = Network(
    feature_extractor=feature_extractor,
    head=heads.DiscreteQNetwork(
        action_dim=num_actions,
        kernel_init=sparse_init,
    ),
)

epsilon_schedule = optax.linear_schedule(1.0, 0.01, int(total_timesteps * 0.2))

agent = QRC(
    cfg=cfg,
    env=env,
    env_params=env_params,
    q_network=q_network,
    h_network=h_network,
    q_optimizer=optax.sgd(1e-4),
    h_optimizer=optax.sgd(1e-5),
    epsilon_schedule=epsilon_schedule,
)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={
                "Algorithm": "QRC",
                "Environment": env_id,
                "Total Timesteps": f"{total_timesteps:_}",
            },
        ),
        WandbLogger(
            project="memorax",
            name="qrc_minatar",
            mode="offline",
            cfg=asdict(cfg),
            seed=seed,
            num_seeds=num_seeds,
        ),
    ]
)

init = jax.vmap(agent.init)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))
evaluate = jax.vmap(lox.spool(agent.evaluate), in_axes=(0, 0, None))

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

    key, eval_key = jax.random.split(key)
    _, logs = evaluate(jax.random.split(eval_key, num_seeds), state, 10_000)
    info = logs.pop("info")
    mask = info["returned_episode"]
    axes = tuple(range(1, mask.ndim))
    episode_returns = jnp.mean(info["returned_episode_returns"], axis=axes, where=mask)
    episode_lengths = jnp.mean(info["returned_episode_lengths"], axis=axes, where=mask)

    data["evaluation/episode_returns"] = episode_returns
    data["evaluation/episode_lengths"] = episode_lengths
    logger.log(data, step=state.step.mean(dtype=jnp.int32).item())

logger.finish()
