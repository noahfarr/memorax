# Core Concepts

## Architecture Overview

Memorax follows a modular architecture where agents are composed of:

```
Agent = Algorithm + Network + Environment
Network = FeatureExtractor -> Torso -> Head
```

## Algorithms

Each algorithm consists of three components:

- **Config**: A frozen dataclass containing hyperparameters
- **State**: A dataclass holding training state (parameters, optimizer state, etc.)
- **Algorithm**: The main class implementing `init()`, `train()`, `warmup()`, and `evaluate()`

```python
from memorax.algorithms import PPO, PPOConfig

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

agent = PPO(config, env, env_params, actor, critic, optimizer, optimizer)
key, state = agent.init(key)
key, state, transitions = agent.train(key, state, num_steps=1000)
```

## Networks

Networks are composed of three parts:

### Feature Extractor

Extracts features from observations, actions, rewards, and done flags:

```python
from memorax.networks import FeatureExtractor
from memorax.networks.layers import Flatten

feature_extractor = FeatureExtractor(
    observation_extractor=Flatten(),
    action_extractor=None,
    reward_extractor=None,
    done_extractor=None,
)
```

### Torso (Sequence Model)

Processes temporal sequences using RNNs, SSMs, or attention:

```python
import flax.linen as nn
from memorax.networks import RNN, Memoroid, MambaCell, SequenceModelWrapper
from memorax.networks.layers import Flatten

torso = RNN(cell=nn.GRUCell(features=64))
torso = Memoroid(cell=MambaCell(features=64))
torso = SequenceModelWrapper(Flatten())
```

### Head

Produces outputs for the RL objective:

```python
from memorax.networks import heads

head = heads.Categorical(action_dim=4)
head = heads.SquashedGaussian(action_dim=2)
head = heads.VNetwork()
head = heads.DiscreteQNetwork(action_dim=4)
```

## JAX Patterns

### Vectorized Environments

All training runs multiple environments in parallel:

```python
config = PPOConfig(num_envs=8, ...)
```

### Random Keys

JAX uses explicit random state management:

```python
key = jax.random.key(0)
key, state = agent.init(key)
key, state, transitions = agent.train(key, state, num_steps=1000)
```

### JIT Compilation

Training loops are JIT-compiled for performance. The first call may be slow due to compilation.

## Transitions

Training produces `Transition` objects containing:

- `first`: The initial timestep
- `second`: The next timestep
- `carry`: Hidden state carry
- `aux`: Auxiliary data

## Buffers

Memorax provides episode-aware replay buffers for off-policy algorithms:

### Episode Buffer

Samples complete sequences while respecting episode boundaries:

```python
from memorax.buffers import make_episode_buffer

buffer = make_episode_buffer(
    max_length=100_000,
    min_length=1000,
    sample_batch_size=32,
    sample_sequence_length=16,
    add_batch_size=8,
)
```

### Prioritized Episode Buffer

Combines episode-aware sampling with Prioritized Experience Replay:

```python
from memorax.buffers import make_prioritised_episode_buffer, compute_importance_weights

buffer = make_prioritised_episode_buffer(
    max_length=100_000,
    min_length=1000,
    sample_batch_size=32,
    sample_sequence_length=16,
    add_batch_size=8,
    priority_exponent=0.6,
)

sample = buffer.sample(state, key)
weights = compute_importance_weights(sample.probabilities, buffer_size, beta=0.4)
state = buffer.set_priorities(state, sample.indices, jnp.abs(td_errors) + 1e-6)
```
