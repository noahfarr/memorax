import time

import flax.linen as nn
import jax
import lox
import optax

from memorax.algorithms import PQN, PQNConfig
from memorax.environments import environment
from memorax.environments.wrappers import RecordEpisodeStatistics, StickyActionWrapper
from memorax.loggers import DashboardLogger, MultiLogger
from memorax.networks import FeatureExtractor, Flatten, Network, heads

total_timesteps = 5_000_000
num_epochs = 10
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 5
num_envs = 128
env_id = "gymnax::Breakout-MinAtar"

env, env_params = environment.make(env_id)
env = StickyActionWrapper(env)
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n

cfg = PQNConfig(
    num_envs=num_envs,
    num_steps=16,
    gamma=0.99,
    td_lambda=0.65,
    num_minibatches=4,
    update_epochs=1,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (
            nn.Conv(16, (3, 3), strides=(1, 1), padding="VALID"),
            nn.relu,
            Flatten(start_dim=-3),
            nn.Dense(128),
            nn.relu,
        )
    ),
)

q_network = Network(
    feature_extractor=feature_extractor,
    head=heads.DiscreteQNetwork(action_dim=num_actions),
)

epsilon_schedule = optax.linear_schedule(1.0, 0.01, int(total_timesteps * 0.2))

agent = PQN(
    cfg=cfg,
    env=env,
    env_params=env_params,
    q_network=q_network,
    optimizer=optax.adam(5e-4),
    epsilon_schedule=epsilon_schedule,
)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={
                "Algorithm": "PQN",
                "Environment": env_id,
                "Total Timesteps": f"{total_timesteps:_}",
            },
        )
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
    episode_returns = info["returned_episode_returns"][info["returned_episode"]]
    episode_lengths = info["returned_episode_lengths"][info["returned_episode"]]

    data = {
        "training/SPS": SPS,
        "training/episode_returns": episode_returns,
        "training/episode_lengths": episode_lengths,
        **logs,
    }

    key, eval_key = jax.random.split(key)
    _, logs = evaluate(jax.random.split(eval_key, num_seeds), state, 1_000)
    info = logs.pop("info")
    episode_returns = info["returned_episode_returns"][info["returned_episode"]]
    episode_lengths = info["returned_episode_lengths"][info["returned_episode"]]

    data["evaluation/episode_returns"] = episode_returns
    data["evaluation/episode_lengths"] = episode_lengths
    logger.log(data, step=state.step.mean().item())

logger.finish()
