import gymnax
from gymnax.environments import environment

from memorax.environments.wrappers import BSuiteWrapper


def make(env_id: str, **kwargs) -> tuple[environment.Environment, environment.EnvParams]:
    env, env_params = gymnax.make(env_id, **kwargs)

    if "bsuite" in env_id:
        env = BSuiteWrapper(env)

    return env, env_params
