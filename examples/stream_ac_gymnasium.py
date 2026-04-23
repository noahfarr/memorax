import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox

from memorax.algorithms import StreamAC, StreamACConfig
from memorax.environments.gymnasium import make
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, MultiLogger, WandbLogger
from memorax.networks import RNN, FeatureExtractor, Network, Residual, Stack, heads
from memorax.networks.blocks.ffn import Projection

total_timesteps = 500_000
num_epochs = 50
num_steps = total_timesteps // num_epochs
seed = 0
num_envs = 16
env_id = "CartPole-v1"

env, env_params = make(env_id, num_envs=num_envs)
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n

config = StreamACConfig(
    num_envs=num_envs,
    trace_lambda=0.8,
    actor_lr=1.0,
    critic_lr=1.0,
    actor_kappa=3.0,
    critic_kappa=2.0,
    entropy_coefficient=0.01,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (nn.Dense(120), nn.relu, nn.Dense(84), nn.relu)
    ),
)

actor_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            Projection(features=128),
            Residual(module=RNN(cell=nn.GRUCell(features=128))),
            Projection(features=64),
        )
    ),
    head=heads.Categorical(action_dim=num_actions),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            Projection(features=128),
            Residual(module=RNN(cell=nn.GRUCell(features=128))),
            Projection(features=64),
        )
    ),
    head=heads.VNetwork(gamma=0.99),
)

agent = StreamAC(config, env, env_params, actor_network, critic_network)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={
                "Algorithm": "AC-Lambda",
                "Environment": env_id,
                "Torso": "GRU",
                "Total Timesteps": f"{total_timesteps:_}",
            },
        ),
        WandbLogger(
            project="memorax",
            name="stream_ac_gymnasium",
            mode="offline",
            cfg=None,
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
