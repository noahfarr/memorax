memorax.networks.heads
======================

Output heads for different RL objectives.

.. currentmodule:: memorax.networks.heads

Policy Heads
------------

:class:`Categorical` - Categorical policy for discrete actions.

:class:`Gaussian` - Gaussian policy for continuous actions.

:class:`SquashedGaussian` - Squashed Gaussian policy (tanh-bounded, used in SAC).

Value Heads
-----------

:class:`VNetwork` - State value function head.

:class:`HLGaussVNetwork` - HL-Gauss value head with two-hot cross-entropy loss.

Q-Network Heads
---------------

:class:`DiscreteQNetwork` - Q-network for discrete actions.

:class:`ContinuousQNetwork` - Q-network for continuous actions.

:class:`TwinContinuousQNetwork` - Twin Q-networks for SAC.

:class:`C51QNetwork` - Categorical DQN with distributional value estimation.

General Value Functions
-----------------------

:class:`GVF` - General Value Function with custom cumulant and discount.

:class:`Horde` - Collection of GVF demons alongside a main head.

Temperature
-----------

:class:`Alpha` - Learnable temperature parameter for SAC.

:class:`Beta` - Learnable temperature parameter.

Other
-----

:class:`PredecessorHead` - Predecessor representation head.

.. autosummary::
   :toctree: generated
   :hidden:

   Categorical
   Gaussian
   SquashedGaussian
   VNetwork
   HLGaussVNetwork
   DiscreteQNetwork
   ContinuousQNetwork
   TwinContinuousQNetwork
   C51QNetwork
   GVF
   Horde
   Alpha
   Beta
   PredecessorHead
