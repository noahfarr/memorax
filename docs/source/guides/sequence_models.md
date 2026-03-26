# Sequence Models

Memorax supports three families of sequence models for memory-augmented RL.

## Overview

| Family | Models | Strengths |
|--------|--------|-----------|
| RNNs | LSTM, GRU, sLSTM, SHM, RTU | Simple, well-understood |
| State Space Models | S5, LRU, Mamba, MinGRU, mLSTM, FFM | Efficient long sequences via parallel scan |
| Attention | SelfAttention, LinearAttention | Flexible, parallel training |

## RNNs

### LSTM / GRU

Standard recurrent networks using Flax cells:

```python
import flax.linen as nn
from memorax.networks import RNN

lstm_torso = RNN(cell=nn.LSTMCell(features=64))
gru_torso = RNN(cell=nn.GRUCell(features=64))
```

### sLSTM

Scalar LSTM with enhanced gating and feature normalization:

```python
from memorax.networks import RNN, sLSTMCell, sLSTMConfig

slstm = RNN(cell=sLSTMCell(config=sLSTMConfig(features=64, hidden_dim=64)))
```

### SHM (Stable Hadamard Memory)

```python
from memorax.networks import RNN, SHMCell, SHMConfig

shm = RNN(cell=SHMCell(config=SHMConfig(features=64, output_features=32)))
```

## State Space Models

All state space models use the `Memoroid` wrapper for parallel scan execution.

### LRU (Linear Recurrent Unit)

```python
from memorax.networks import Memoroid, LRUCell, LRUConfig

lru = Memoroid(cell=LRUCell(config=LRUConfig(features=64, hidden_dim=64)))
```

### S5 (Simplified Structured State Space)

```python
from memorax.networks import Memoroid, S5Cell

s5 = Memoroid(cell=S5Cell(config=S5Config(features=64, hidden_dim=64)))
```

### Mamba (Selective State Space Model)

```python
from memorax.networks import Memoroid, Mamba2Cell, Mamba2Config

mamba = Memoroid(cell=Mamba2Cell(config=Mamba2Config(features=64, num_heads=4, head_dim=16)))
```

### MinGRU

Minimal GRU variant computed in log-space for numerical stability:

```python
from memorax.networks import Memoroid, MinGRUCell, MinGRUConfig

mingru = Memoroid(cell=MinGRUCell(config=MinGRUConfig(features=64)))
```

### mLSTM (Matrix LSTM)

Matrix LSTM using gated linear attention:

```python
from memorax.networks import Memoroid, mLSTMCell, mLSTMConfig

mlstm = Memoroid(cell=mLSTMCell(config=mLSTMConfig(features=64, hidden_dim=64, num_heads=4)))
```

### FFM (Fast and Forgetful Memory)

```python
from memorax.networks import Memoroid, FFMCell, FFMConfig

ffm = Memoroid(cell=FFMCell(config=FFMConfig(features=64, memory_size=32, context_size=64)))
```

### RTU (Rotational Transformation Unit)

```python
from memorax.networks import RNN, RTUCell, RTUConfig

rtu = RNN(cell=RTUCell(config=RTUConfig(features=64, hidden_dim=64)))
```

## Attention

### Self-Attention

Multi-head self-attention (used directly, no wrapper needed):

```python
from memorax.networks import SelfAttention, SelfAttentionConfig

attention = SelfAttention(config=SelfAttentionConfig(features=64, num_heads=4, context_length=128))
```

### Linear Attention

Efficient linear-complexity attention via kernelized features:

```python
from memorax.networks import Memoroid, LinearAttentionCell, LinearAttentionConfig

linear_attention = Memoroid(cell=LinearAttentionCell(config=LinearAttentionConfig(features=64, num_heads=4, head_dim=16)))
```

## RTRL (Real-Time Recurrent Learning)

Wraps any sequence model to compute real-time gradients through the recurrence:

```python
from memorax.networks import RTRL, RNN
import flax.linen as nn

rtrl_gru = RTRL(model=RNN(cell=nn.GRUCell(features=64)))
```

## RL2 Wrapper

Preserves hidden state across episode boundaries within a trial for meta-RL:

```python
from memorax.networks import RL2Wrapper, RNN
import flax.linen as nn

rl2 = RL2Wrapper(model=RNN(cell=nn.GRUCell(features=64)))
```

## Choosing a Model

### For Short Episodes (< 100 steps)
- **LSTM/GRU**: Simple and effective
- **sLSTM**: Enhanced gating

### For Long Episodes (100-1000 steps)
- **S5/LRU**: Efficient state space models
- **Mamba**: Selective attention to inputs
- **RTU**: Efficient rotational dynamics

### For Very Long Episodes (> 1000 steps)
- **SelfAttention**: With positional embeddings
- **LinearAttention**: Linear complexity

### For Memory-Intensive Tasks
- **FFM/SHM**: Explicit memory mechanisms
- **mLSTM**: Matrix memory

## Example: Mamba Agent

```python
from memorax.algorithms import PPO, PPOConfig
from memorax.networks import Network, FeatureExtractor, Memoroid, Mamba2Cell, Mamba2Config, heads
from memorax.networks.layers import Flatten

actor = Network(
    feature_extractor=FeatureExtractor(observation_extractor=Flatten()),
    torso=Memoroid(cell=Mamba2Cell(config=Mamba2Config(features=64, num_heads=4, head_dim=16))),
    head=heads.Categorical(action_dim=4),
)

critic = Network(
    feature_extractor=FeatureExtractor(observation_extractor=Flatten()),
    torso=Memoroid(cell=Mamba2Cell(config=Mamba2Config(features=64, num_heads=4, head_dim=16))),
    head=heads.VNetwork(),
)

agent = PPO(config, env, env_params, actor, critic, optimizer, optimizer)
```
