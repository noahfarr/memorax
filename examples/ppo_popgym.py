import time
from dataclasses import asdict

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax

from memorax.algorithms import PPO, PPOConfig
from memorax.environments.popgym import make
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, MultiLogger, WandbLogger
from memorax.networks import RNN, FeatureExtractor, Network, Stack, heads
from memorax.networks.blocks.ffn import Projection

total_timesteps = 15_000_000
num_epochs = 150
num_steps = total_timesteps // num_epochs
seed = 0
num_envs = 64
env_id = "popgym-RepeatPreviousEasy-v0"

env, env_params = make(env_id, batch_shape=(num_envs,))
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n
num_obs = int(env.observation_high) + 1

cfg = PPOConfig(
    num_envs=num_envs,
    num_steps=1024,
    gamma=0.99,
    gae_lambda=1.0,
    num_minibatches=8,
    update_epochs=30,
    normalize_advantage=True,
    clip_coefficient=0.3,
    clip_value_loss=True,
    entropy_coefficient=0.0,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (
            partial(jax.nn.one_hot, num_classes=num_obs),
            nn.Dense(256),
            nn.LayerNorm(),
            nn.leaky_relu,
        )
    ),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(5e-5),
)

actor_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            RNN(cell=nn.GRUCell(features=256)),
            Projection(features=256),
        )
    ),
    head=heads.Categorical(action_dim=num_actions),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            RNN(cell=nn.GRUCell(features=256)),
            Projection(features=256),
        )
    ),
    head=heads.VNetwork(),
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
            name="ppo_popgym",
            mode="offline",
            cfg=asdict(cfg),
            seed=seed,
            num_seeds=1,
        ),
    ]
)

init = jax.jit(agent.init)
train = lox.spool(jax.jit(agent.train, static_argnames=["num_steps"]))

key = jax.random.key(seed)

key, init_key = jax.random.split(key)
state = init(init_key)

for i in range(num_epochs):
    start = time.perf_counter()
    key, train_key = jax.random.split(key)
    state, logs = train(train_key, state, num_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_steps / (end - start))

    info = logs.pop("info")
    mask = info["returned_episode"]
    episode_returns = jnp.mean(info["returned_episode_returns"], where=mask)
    episode_lengths = jnp.mean(info["returned_episode_lengths"], where=mask)

    data = {
        "training/SPS": SPS,
        "training/episode_returns": episode_returns,
        "training/episode_lengths": episode_lengths,
        **logs,
    }
    logger.log(data, step=state.step.item())

logger.finish()
