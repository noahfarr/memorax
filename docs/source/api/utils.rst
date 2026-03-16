memorax.utils
=============

Utility functions and data structures.

.. currentmodule:: memorax.utils

Data Structures
---------------

:class:`Timestep` - Environment timestep container.

:class:`Transition` - Transition data for training.

Functions
---------

:func:`periodic_incremental_update` - Polyak averaging for target networks.

:func:`delayed_update` - Delayed parameter updates.

:func:`callback` - Decorator for debug callbacks.

.. autosummary::
   :toctree: generated
   :hidden:

   Timestep
   Transition
   periodic_incremental_update
   delayed_update
   callback
