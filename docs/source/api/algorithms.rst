memorax.algorithms
==================

Reinforcement learning algorithms for training agents.

.. currentmodule:: memorax.algorithms

PPO
---

:class:`PPO` - Proximal Policy Optimization for discrete and continuous action spaces.

:class:`PPOConfig` - Configuration dataclass for PPO.

:class:`PPOState` - Training state for PPO.

MAPPO
-----

:class:`MAPPO` - Multi-Agent PPO for multi-agent environments.

:class:`MAPPOConfig` - Configuration dataclass for MAPPO.

:class:`MAPPOState` - Training state for MAPPO.

DQN
---

:class:`DQN` - Deep Q-Network with double Q-learning.

:class:`DQNConfig` - Configuration dataclass for DQN.

:class:`DQNState` - Training state for DQN.

R2D2
----

:class:`R2D2` - Recurrent Experience Replay in Distributed RL.

:class:`R2D2Config` - Configuration dataclass for R2D2.

:class:`R2D2State` - Training state for R2D2.

SAC
---

:class:`SAC` - Soft Actor-Critic for continuous control.

:class:`SACConfig` - Configuration dataclass for SAC.

:class:`SACState` - Training state for SAC.

PQN
---

:class:`PQN` - Parallelised Q-Network (on-policy Q-learning).

:class:`PQNConfig` - Configuration dataclass for PQN.

:class:`PQNState` - Training state for PQN.

ACLambda
--------

:class:`ACLambda` - Actor-Critic with eligibility traces.

:class:`ACLambdaConfig` - Configuration dataclass for ACLambda.

:class:`ACLambdaState` - Training state for ACLambda.

GradientPPO
------------

:class:`GradientPPO` - PPO with gradient eligibility traces.

:class:`GradientPPOConfig` - Configuration dataclass for GradientPPO.

:class:`GradientPPOState` - Training state for GradientPPO.

.. autosummary::
   :toctree: generated
   :hidden:

   PPO
   PPOConfig
   PPOState
   MAPPO
   MAPPOConfig
   MAPPOState
   DQN
   DQNConfig
   DQNState
   R2D2
   R2D2Config
   R2D2State
   SAC
   SACConfig
   SACState
   PQN
   PQNConfig
   PQNState
   ACLambda
   ACLambdaConfig
   ACLambdaState
   GradientPPO
   GradientPPOConfig
   GradientPPOState
