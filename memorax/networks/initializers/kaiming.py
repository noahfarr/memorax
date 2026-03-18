from flax import linen as nn


def kaiming_uniform() -> nn.initializers.Initializer:
    return nn.initializers.variance_scaling(2.0 / (1 + 5), "fan_in", "uniform")
