# Installation

## Basic Installation

Install Memorax using pip:

```bash
pip install memorax
```

Or using uv:

```bash
uv add memorax
```

## Optional Dependencies

Memorax supports various environment backends. Install the ones you need:

```bash
pip install "memorax[brax]"
pip install "memorax[craftax]"
pip install "memorax[navix]"
pip install "memorax[popgym-arcade]"
pip install "memorax[popjym]"
pip install "memorax[gxm]"
pip install "memorax[jaxmarl]"
pip install "memorax[xminigrid]"
pip install "memorax[playground]"
```

Or install a bundled environment stack:

```bash
pip install "memorax[environments]"
```

## GPU Support

For GPU acceleration with CUDA:

```bash
pip install "memorax[cuda]"
```

## Development Installation

To contribute to Memorax:

```bash
git clone https://github.com/memory-rl/memorax.git
cd memorax
uv sync
uv run pre-commit install
```

## Verifying Installation

```python
import memorax
from memorax.algorithms import PPO
from memorax.environments import make

env, env_params = make("gymnax::CartPole-v1")
print(f"Memorax version: {memorax.__version__}")
print(f"Environment created: {env}")
```
