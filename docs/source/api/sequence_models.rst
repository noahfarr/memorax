memorax.networks.sequence_models
================================

Sequence models for temporal processing.

.. currentmodule:: memorax.networks

RNN Models
----------

:class:`RNN` - Wrapper for Flax RNN cells (LSTM, GRU, etc.).

:class:`sLSTMCell` - Scalar LSTM cell with enhanced gating and feature normalization.

:class:`SHMCell` - Stable Hadamard Memory cell.

Memoroid Models
---------------

:class:`Memoroid` - Wrapper for parallel-scannable sequence models using associative scan.

:class:`MemoroidCellBase` - Base class for memoroid cells.

:class:`Mamba2Cell` - Selective State Space Model cell (Mamba-2).

:class:`Mamba3Cell` - State Space Model with trapezoidal discretization and complex state (Mamba-3).

:class:`S5Cell` - Simplified Structured State Space cell.

:class:`LRUCell` - Linear Recurrent Unit cell.

:class:`MinGRUCell` - Minimal GRU cell (log-space).

:class:`mLSTMCell` - Matrix LSTM cell with gated linear attention.

:class:`FFMCell` - Fast and Forgetful Memory cell.

:class:`LinearAttentionCell` - Linear attention cell with kernelized features.

:class:`RTUCell` - Rotational Transformation Unit cell.

Attention
---------

:class:`SelfAttention` - Multi-head self-attention with optional cross-segment memory.

Wrappers
--------

:class:`SequenceModelWrapper` - Wraps non-recurrent models as sequence models.

:class:`RL2Wrapper` - RL² wrapper that preserves hidden state across episode boundaries.

:class:`RTRL` - Real-Time Recurrent Learning wrapper for online gradient computation.

.. autosummary::
   :toctree: generated
   :hidden:

   RNN
   sLSTMCell
   SHMCell
   Memoroid
   MemoroidCellBase
   Mamba2Cell
   Mamba3Cell
   S5Cell
   LRUCell
   MinGRUCell
   mLSTMCell
   FFMCell
   LinearAttentionCell
   RTUCell
   SelfAttention
   SequenceModelWrapper
   RL2Wrapper
   RTRL
