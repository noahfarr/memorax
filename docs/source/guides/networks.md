# Building Networks

A Memorax network is a pipeline of three components: a **feature extractor** that maps raw observations into a feature vector, a **torso** that processes temporal sequences (RNNs, SSMs, attention), and a **head** that produces the final output (action distribution, Q-values, or state value).

```python
import flax.linen as nn
from memorax.networks import FeatureExtractor, Network, RNN, heads

actor = Network(
    feature_extractor=FeatureExtractor(
        observation_extractor=nn.Sequential((nn.Dense(64), nn.relu)),
    ),
    torso=RNN(cell=nn.GRUCell(features=64)),
    head=heads.Categorical(action_dim=4),
)
```

Each of these three components is a Flax module, so you can swap them freely. The rest of this guide walks through what's available for each slot.

## Feature Extractors

The `FeatureExtractor` takes the raw environment output and turns it into a flat feature vector. At minimum you need an `observation_extractor`, but you can also include extractors for actions, rewards, and done flags. These get concatenated together, which is useful for architectures like RL² that condition on the full transition:

```python
feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential((nn.Dense(64), nn.relu)),
    action_extractor=nn.Sequential((nn.Dense(32), nn.relu)),
    reward_extractor=nn.Sequential((nn.Dense(16), nn.relu)),
    done_extractor=nn.Sequential((nn.Dense(16), nn.relu)),
)
```

Any Flax module works here. For image observations you might use a CNN or a `ViT` with `PatchEmbedding`:

```python
from memorax.networks import ViT, PatchEmbedding

feature_extractor = FeatureExtractor(
    observation_extractor=ViT(
        features=64, num_layers=4, num_heads=4, expansion_factor=4,
        patch_embedding=PatchEmbedding(patch_size=8, features=64),
    ),
)
```

## Torsos

The torso sits between the feature extractor and the head, and is responsible for temporal modeling. Memorax provides three families of sequence models that can be used directly as torsos.

**RNNs** wrap standard Flax recurrent cells. They process sequences step-by-step and maintain a hidden state carry:

```python
torso = RNN(cell=nn.GRUCell(features=64))
torso = RNN(cell=nn.LSTMCell(features=64))
```

Memorax also provides custom RNN cells like `sLSTMCell` and `SHMCell` that work the same way.

**Memoroid models** use JAX's associative scan for efficient O(T log T) parallel training. They cover state space models and linear recurrences. You wrap a cell in `Memoroid`:

```python
from memorax.networks import Memoroid, Mamba2Cell, Mamba2Config, S5Cell, S5Config, LRUCell, LRUConfig, mLSTMCell, mLSTMConfig, MinGRUCell, MinGRUConfig, FFMCell, FFMConfig

torso = Memoroid(cell=Mamba2Cell(config=Mamba2Config(features=64)))
torso = Memoroid(cell=S5Cell(config=S5Config(features=64, hidden_dim=64)))
torso = Memoroid(cell=LRUCell(config=LRUConfig(features=64, hidden_dim=64)))
torso = Memoroid(cell=mLSTMCell(config=mLSTMConfig(features=64, hidden_dim=64, num_heads=4)))
```

**Attention** models like `SelfAttention` can be used directly as a torso. For linear-complexity attention, wrap `LinearAttentionCell` in a `Memoroid`:

```python
from memorax.networks import SelfAttention, SelfAttentionConfig, LinearAttentionCell, LinearAttentionConfig

torso = SelfAttention(config=SelfAttentionConfig(features=64, num_heads=4, context_length=128))
torso = Memoroid(cell=LinearAttentionCell(config=LinearAttentionConfig(features=64, num_heads=4, head_dim=16)))
```

If you want to use a non-recurrent module as a torso (for example a simple MLP), wrap it with `SequenceModelWrapper` so it conforms to the sequence model interface:

```python
from memorax.networks import SequenceModelWrapper

torso = SequenceModelWrapper(nn.Sequential((nn.Dense(64), nn.relu)))
```

See the {doc}`sequence_models` guide for a deeper comparison of when to use each model family.

## Heads

The head takes the torso's output and produces whatever the algorithm needs. For **discrete action spaces**, use `Categorical` for policy gradient methods or `DiscreteQNetwork` for value-based methods. `C51QNetwork` adds distributional value estimation. For **continuous action spaces**, `Gaussian` and `SquashedGaussian` (tanh-bounded, used in SAC) produce action distributions, while `ContinuousQNetwork` and `TwinContinuousQNetwork` (twin Q for SAC) output Q-values. For **critics**, `VNetwork` is the standard value head, and `HLGaussVNetwork` provides a distributional alternative using two-hot cross-entropy.

```python
from memorax.networks import heads

# Discrete
heads.Categorical(action_dim=4)
heads.DiscreteQNetwork(action_dim=4)

# Continuous
heads.SquashedGaussian(action_dim=2)
heads.TwinContinuousQNetwork()

# Value
heads.VNetwork()
heads.HLGaussVNetwork(num_bins=101)
```

For auxiliary prediction tasks, `GVF` defines a general value function with a custom cumulant and discount, and `Horde` bundles multiple GVF demons alongside a main head.

## Composable Blocks

For deeper architectures you can compose layers using blocks for normalization, residual connections, and gating. These snap together to build architectures like GTrXL or xLSTM from modular pieces. Here's a GTrXL-style transformer torso:

```python
from memorax.networks import (
    FFN, ALiBi, GatedResidual, PreNorm,
    SegmentRecurrence, SelfAttention, SelfAttentionConfig, Stack,
)

features, num_heads, num_layers = 64, 4, 2

attention = GatedResidual(PreNorm(SegmentRecurrence(
    SelfAttention(config=SelfAttentionConfig(features=features, num_heads=num_heads, context_length=128, positional_embedding=ALiBi(num_heads))),
    memory_length=64, features=features,
)))
ffn = GatedResidual(PreNorm(FFN(features=features, expansion_factor=4)))
torso = Stack(blocks=(attention, ffn) * num_layers)
```

Available blocks include `Residual`, `GatedResidual`, `PreNorm`, `PostNorm`, `FFN`, `GLU`, `Projection`, `SegmentRecurrence`, `Stack`, `MoE`, and `TopKRouter`. Positional embeddings (`RoPE`, `ALiBi`, `LearnablePositionalEmbedding`) can be passed to attention layers.
