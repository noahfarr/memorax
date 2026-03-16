# Building Networks

This guide covers how to build neural networks for RL agents in Memorax.

## Network Architecture

A Memorax network consists of three components:

```
Input -> FeatureExtractor -> Torso -> Head -> Output
```

## Feature Extractor

The `FeatureExtractor` processes raw inputs into feature vectors:

```python
from memorax.networks import FeatureExtractor
from memorax.networks.layers import Flatten

feature_extractor = FeatureExtractor(
    observation_extractor=Flatten(),
)

feature_extractor = FeatureExtractor(
    observation_extractor=Flatten(),
    action_extractor=Flatten(),
    reward_extractor=Flatten(),
    done_extractor=Flatten(),
)
```

## Torso (Sequence Models)

The torso processes temporal sequences. Non-recurrent modules need `SequenceModelWrapper`:

```python
import flax.linen as nn
from memorax.networks import (
    RNN, Memoroid, SelfAttention, SequenceModelWrapper,
    MambaCell, S5Cell, LRUCell, MinGRUCell, mLSTMCell,
)
from memorax.networks.layers import Flatten

torso = RNN(cell=nn.GRUCell(features=64))
torso = RNN(cell=nn.LSTMCell(features=64))
torso = Memoroid(cell=MambaCell(features=64))
torso = Memoroid(cell=S5Cell(features=64, state_dim=64))
torso = SelfAttention(features=64, num_heads=4, head_dim=16)
torso = SequenceModelWrapper(Flatten())
```

## Heads

Heads produce outputs for different RL objectives:

### Discrete Actions

```python
from memorax.networks import heads

actor_head = heads.Categorical(action_dim=4)
q_head = heads.DiscreteQNetwork(action_dim=4)
c51_head = heads.C51QNetwork(action_dim=4, num_atoms=51)
```

### Continuous Actions

```python
actor_head = heads.Gaussian(action_dim=2)
actor_head = heads.SquashedGaussian(action_dim=2)
q_head = heads.ContinuousQNetwork()
twin_q_head = heads.TwinContinuousQNetwork()
```

### Value Functions

```python
critic_head = heads.VNetwork()
hlgauss_head = heads.HLGaussVNetwork(num_bins=101, v_min=-10.0, v_max=10.0)
```

### General Value Functions

```python
gvf = heads.GVF(head=heads.VNetwork(), gamma=0.99, cumulant=my_cumulant_fn)
horde = heads.Horde(head=heads.VNetwork(), demons={"task1": gvf1, "task2": gvf2})
```

## Composing Networks

Use the `Network` class to compose components:

```python
from memorax.networks import Network, FeatureExtractor, RNN, heads
from memorax.networks.layers import Flatten
import flax.linen as nn

actor = Network(
    feature_extractor=FeatureExtractor(observation_extractor=Flatten()),
    torso=RNN(cell=nn.GRUCell(features=64)),
    head=heads.Categorical(action_dim=4),
)

critic = Network(
    feature_extractor=FeatureExtractor(observation_extractor=Flatten()),
    torso=RNN(cell=nn.GRUCell(features=64)),
    head=heads.VNetwork(),
)
```

## Using Blocks

Add architectural blocks for more complex networks:

```python
from memorax.networks import SelfAttention, FFN, PreNorm, Residual, GatedResidual, GLU, Stack

torso = Stack(blocks=[
    PreNorm(Residual(SelfAttention(features=64, num_heads=4, head_dim=16))),
    PreNorm(Residual(FFN(hidden_dim=256))),
    PreNorm(GatedResidual(GLU(hidden_dim=256))),
])
```

## Vision Transformer

For image-based observations:

```python
from memorax.networks import ViT, PatchEmbedding

encoder = ViT(
    features=64,
    num_layers=4,
    num_heads=4,
    expansion_factor=4,
    patch_embedding=PatchEmbedding(patch_size=8, features=64),
)
```

## Positional Embeddings

```python
from memorax.networks import RoPE, ALiBi, LearnablePositionalEmbedding

rope = RoPE()
alibi = ALiBi()
wpe = LearnablePositionalEmbedding(max_length=512, features=64)
```
