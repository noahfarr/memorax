memorax.buffers
===============

Episode-aware replay buffers for off-policy algorithms.

.. currentmodule:: memorax.buffers

Episode Buffer
--------------

.. autofunction:: make_episode_buffer

.. autofunction:: get_full_start_flags

.. autofunction:: get_start_flags_from_done

Prioritized Episode Buffer
---------------------------

.. autofunction:: make_prioritised_episode_buffer

.. autofunction:: compute_importance_weights

.. autoclass:: PrioritisedEpisodeBufferSample
   :members:
