import time

import flax.linen as nn
import jax
import lox
import optax

from memorax.algorithms import QRC, QRCConfig
from memorax.environments import environment
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, MultiLogger
from memorax.networks import FeatureExtractor, Flatten, Network, heads
from memorax.networks.initializers import sparse

total_timesteps = 5_000_000
num_epochs = 50
num_steps = total_timesteps // num_epochs
seed = 0
num_envs = 16
env_id = "gymnax::Breakout-MinAtar"

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)

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
        )
    ]
)

init = jax.jit(agent.init)
warmup = jax.jit(agent.warmup, static_argnames=["num_steps"])
train = lox.spool(jax.jit(agent.train, static_argnames=["num_steps"]))

key = jax.random.key(seed)

key, init_key = jax.random.split(key)
state = init(init_key)
key, warmup_key = jax.random.split(key)
state = warmup(warmup_key, state, 5_000)

for i in range(num_epochs):
    start = time.perf_counter()
    key, train_key = jax.random.split(key)
    state, logs = train(train_key, state, num_steps)
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
    logger.log(data, step=state.step.item())

logger.finish()
