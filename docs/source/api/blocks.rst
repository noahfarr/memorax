memorax.networks.blocks
=======================

Building blocks for constructing network architectures.

.. currentmodule:: memorax.networks.blocks

Feed-Forward
------------

:class:`FFN` - Feed-forward network block with expansion.

:class:`GLU` - Gated Linear Unit variant of FFN.

:class:`Projection` - Single linear projection layer.

Normalization
-------------

:class:`PreNorm` - Pre-normalization wrapper.

:class:`PostNorm` - Post-normalization wrapper.

Residual
--------

:class:`Residual` - Residual connection wrapper.

:class:`GatedResidual` - Gated residual connection with learned gate.

Composition
-----------

:class:`Stack` - Stacks multiple blocks sequentially.

:class:`SegmentRecurrence` - Fixed-length cross-segment memory buffer.

Mixture of Experts
------------------

:class:`MoE` - Mixture of Experts layer.

:class:`TopKRouter` - Top-K routing for MoE.

.. autosummary::
   :toctree: generated
   :hidden:

   FFN
   GLU
   Projection
   PreNorm
   PostNorm
   Residual
   GatedResidual
   Stack
   SegmentRecurrence
   MoE
   TopKRouter
