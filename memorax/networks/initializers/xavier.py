from flax import linen as nn


def xavier_uniform():
    return nn.initializers.variance_scaling(1.0, "fan_avg", "uniform")
