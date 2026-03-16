memorax.networks.layers
=======================

Low-level layer primitives.

.. currentmodule:: memorax.networks

Convolution
-----------

:class:`CausalConv1d` - Stateful causal 1D convolution for recurrent use.

:class:`ParallelCausalConv1d` - Parallel (non-recurrent) causal convolution.

Dense
-----

:class:`BlockDiagonalDense` - Block-diagonal dense layer for efficient computation.

Normalization
-------------

:class:`MultiHeadLayerNorm` - Per-head layer normalization.

Utility
-------

:class:`Flatten` - Reshape to batch × feature.

:class:`Identity` - Pass-through layer.

.. autosummary::
   :toctree: generated
   :hidden:

   CausalConv1d
   ParallelCausalConv1d
   BlockDiagonalDense
   MultiHeadLayerNorm
   Flatten
   Identity
