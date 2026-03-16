# Memorax

```{raw} html
<div class="hero-section">
  <h1>Memorax</h1>
  <p>A unified JAX/Flax framework for memory-augmented reinforcement learning with RNNs, SSMs, and Transformers.</p>
</div>

<div class="cta-grid">
  <a class="cta-card" href="getting_started/installation.html">
    <h3>Install</h3>
    <p>Get up and running with pip or uv in seconds.</p>
  </a>
  <a class="cta-card" href="getting_started/quickstart.html">
    <h3>Quick Start</h3>
    <p>Train your first memory-augmented agent.</p>
  </a>
  <a class="cta-card" href="api/index.html">
    <h3>API Reference</h3>
    <p>Complete reference for all modules and classes.</p>
  </a>
</div>
```

## Features

```{raw} html
<div class="feature-grid">
  <div class="feature-card">
    <h3>Algorithms</h3>
    <p>JAX implementations of DQN, PPO, MAPPO, SAC, PQN, R2D2, AC(Lambda), and GradientPPO &mdash; all with burn-in support for recurrent networks.</p>
  </div>
  <div class="feature-card">
    <h3>Sequence Models</h3>
    <p>LSTM, GRU, sLSTM, mLSTM, S5, LRU, Mamba, MinGRU, FFM, SHM, RTU, plus Self-Attention and Linear Attention. Compose GPT-2, GTrXL, or xLSTM-style architectures from modular blocks.</p>
  </div>
  <div class="feature-card">
    <h3>Modular Networks</h3>
    <p>MLP, CNN, and ViT encoders with RoPE and ALiBi positional embeddings. Mixture of Experts for horizontal scaling. Full feature extractor &rarr; torso &rarr; head composition.</p>
  </div>
  <div class="feature-card">
    <h3>Parallel Scan Algebra</h3>
    <p>Memoroid cells use JAX associative scan for efficient O(T log T) parallel training of recurrent models. RTRL support for online gradient computation.</p>
  </div>
  <div class="feature-card">
    <h3>Environment Integration</h3>
    <p>Plug-and-play with Gymnax, Brax, Navix, Craftax, MuJoCo Playground, POPGym, XMiniGrid, JaxMARL, and more through a unified <code>make()</code> interface.</p>
  </div>
  <div class="feature-card">
    <h3>Logging</h3>
    <p>Composable logging with W&amp;B, TensorBoard, Neptune, file, and terminal dashboard backends. Aggregate multiple loggers with a single API.</p>
  </div>
</div>
```

## Ecosystem

```{raw} html
<div class="ecosystem-grid">
  <div class="ecosystem-badge">Gymnax</div>
  <div class="ecosystem-badge">Brax</div>
  <div class="ecosystem-badge">Navix</div>
  <div class="ecosystem-badge">Craftax</div>
  <div class="ecosystem-badge">MuJoCo</div>
  <div class="ecosystem-badge">POPGym</div>
  <div class="ecosystem-badge">POPJym</div>
  <div class="ecosystem-badge">XMiniGrid</div>
  <div class="ecosystem-badge">JaxMARL</div>
  <div class="ecosystem-badge">GXM</div>
  <div class="ecosystem-badge">Grimax</div>
</div>
```

## Quick Example

```python
import jax
import optax
from memorax.algorithms import PPO, PPOConfig
from memorax.environments import make
from memorax.networks import FeatureExtractor, Network, heads
from memorax.networks.layers import Flatten

env, env_params = make("gymnax::CartPole-v1")

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

feature_extractor = FeatureExtractor(observation_extractor=Flatten())
actor = Network(feature_extractor, head=heads.Categorical(env.action_space(env_params).n))
critic = Network(feature_extractor, head=heads.VNetwork())
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))

agent = PPO(config, env, env_params, actor, critic, optimizer, optimizer)
key, state = agent.init(jax.random.key(0))
key, state, transitions = agent.train(key, state, num_steps=10_000)
```

## Citation

```bibtex
@software{memoryrl2025github,
  title   = {Memory-RL: A Unified Framework for Memory-Augmented Reinforcement Learning},
  author  = {Noah Farr},
  year    = {2025},
  url     = {https://github.com/memory-rl/memorax}
}
```

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

getting_started/installation
getting_started/quickstart
getting_started/concepts
```

```{toctree}
:maxdepth: 2
:caption: User Guides
:hidden:

guides/algorithms
guides/networks
guides/sequence_models
```

```{toctree}
:maxdepth: 3
:caption: API Reference
:hidden:

api/index
```
