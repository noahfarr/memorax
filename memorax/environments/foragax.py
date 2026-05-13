from gymnax.environments import environment


def make(env_id: str, **kwargs) -> tuple[environment.Environment, environment.EnvParams]:
    from foragax.registry import make as foragax_make

    env = foragax_make(env_id, **kwargs)
    return env, env.default_params
