import jax.numpy as jnp

from memorax.environments.wrappers import GymnaxWrapper


def make(env_id, **kwargs):
    import jaxmarl

    env = jaxmarl.make(env_id, **kwargs)
    env = JaxMarlGymnaxWrapper(env)
    return env, None


class JaxMarlGymnaxWrapper(GymnaxWrapper):

    @property
    def agents(self):
        return self._env.agents

    @property
    def num_agents(self):
        return len(self._env.agents)

    @property
    def action_spaces(self, params=None):
        return self._env.action_spaces

    def reset(self, key, params=None):
        obs_dict, state = self._env.reset(key)
        obs = jnp.stack([obs_dict[aid] for aid in self.agents])
        return obs, state

    def step(self, key, state, actions, params=None):
        actions_dict = {aid: actions[i] for i, aid in enumerate(self.agents)}
        obs_dict, state, reward_dict, done_dict, info = self._env.step(
            key, state, actions_dict
        )
        return (
            jnp.stack([obs_dict[aid] for aid in self.agents]),
            state,
            jnp.stack([reward_dict[aid] for aid in self.agents]),
            jnp.stack([done_dict[aid] for aid in self.agents]),
            info,
        )
