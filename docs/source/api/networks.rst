memorax.networks
================

Neural network components for building RL agents.

.. currentmodule:: memorax.networks

Core
----

:class:`Network` - Main network class composing feature extractors, torsos, and heads.

:class:`FeatureExtractor` - Extracts features from observations, actions, rewards, and done flags.

:class:`Identity` - Identity module that passes input through unchanged.

Vision
------

:class:`ViT` - Vision Transformer.

:class:`PatchEmbedding` - Converts images to patch sequences via Conv2D.

Wrappers
--------

:class:`SequenceModelWrapper` - Wraps non-recurrent models for use as sequence models.

.. autosummary::
   :toctree: generated
   :hidden:

   Network
   FeatureExtractor
   Identity
   ViT
   PatchEmbedding
   SequenceModelWrapper
