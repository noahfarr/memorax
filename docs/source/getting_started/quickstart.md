# Quick Start

This guide demonstrates how to train a PPO agent with GRU memory on CartPole.

```python
from dataclasses import asdict

import flax.linen as nn
import jax
import lox
import optax

from memorax.algorithms import PPO, PPOConfig
from memorax.environments import make
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import FeatureExtractor, Network, RNN, heads
from memorax.networks.layers import Flatten

env, env_params = make("gymnax::CartPole-v1")
env = RecordEpisodeStatistics(env)

config = PPOConfig(
    num_envs=8,
    num_steps=128,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
    normalize_advantage=True,
    clip_coefficient=0.2,
    clip_value_loss=True,
    entropy_coefficient=0.01,
)

d_model = 64

feature_extractor = FeatureExtractor(
    observation_extractor=Flatten(),
)
torso = RNN(cell=nn.GRUCell(features=d_model))
actor_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.Categorical(
        action_dim=env.action_space(env_params).n,
        kernel_init=nn.initializers.orthogonal(scale=0.01),
    ),
)
critic_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.VNetwork(kernel_init=nn.initializers.orthogonal(scale=1.0)),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)

agent = PPO(
    config=config,
    env=env,
    env_params=env_params,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
)
logger = Logger([DashboardLogger(title="PPO-GRU CartPole", total_timesteps=500_000)])
logger_state = logger.init(cfg=asdict(config))

init = jax.vmap(agent.init)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

key = jax.random.key(0)
keys = jax.random.split(key, 1)

keys, state = init(keys)

for i in range(0, 500_000, 10_000):
    (keys, state), logs = train(keys, state, 10_000)

    info = logs.pop("info")
    episode_returns = info["returned_episode_returns"][info["returned_episode"]]
    episode_lengths = info["returned_episode_lengths"][info["returned_episode"]]

    data = {
        "training/episode_returns": episode_returns,
        "training/episode_lengths": episode_lengths,
        **logs,
    }
    logger_state = logger.log(logger_state, data, step=state.step[0].item())
    logger.emit(logger_state)

logger.finish(logger_state)
```

## Next Steps

- Learn about different {doc}`../guides/algorithms`
- Explore available {doc}`../guides/sequence_models`
- Build custom {doc}`../guides/networks`
