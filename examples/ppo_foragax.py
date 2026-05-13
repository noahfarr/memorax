import time
from dataclasses import asdict

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax

from memorax.algorithms import PPO, PPOConfig
from memorax.environments import environment
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, MultiLogger, WandbLogger
from memorax.networks import RNN, FeatureExtractor, Flatten, Network, Stack, heads
from memorax.networks.blocks.ffn import Projection

total_timesteps = 5_000_000
num_epochs = 50
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "foragax::ForagaxTwoBiomeLarge-v1"

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n

cfg = PPOConfig(
    num_envs=64,
    num_steps=128,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
    normalize_advantage=True,
    clip_coefficient=0.2,
    clip_value_loss=True,
    entropy_coefficient=0.01,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (
            nn.Conv(32, (3, 3), strides=(1, 1), padding="SAME"),
            nn.relu,
            Flatten(start_dim=-3),
            nn.Dense(128),
            nn.relu,
        )
    ),
)

actor_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            RNN(cell=nn.GRUCell(features=128)),
            Projection(features=128),
        )
    ),
    head=heads.Categorical(action_dim=num_actions),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            RNN(cell=nn.GRUCell(features=128)),
            Projection(features=128),
        )
    ),
    head=heads.VNetwork(),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(3e-4),
)

agent = PPO(cfg, env, env_params, actor_network, critic_network, optimizer, optimizer)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={
                "Algorithm": "PPO",
                "Environment": env_id,
                "Torso": "GRU",
                "Total Timesteps": f"{total_timesteps:_}",
            },
        ),
        WandbLogger(
            project="memorax",
            name="ppo_foragax",
            mode="offline",
            cfg=asdict(cfg),
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
