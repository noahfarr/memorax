import functools
from typing import Callable

import jax


def callback(function: Callable) -> Callable:
    @functools.wraps(function)
    def wrapper(*args, **kwargs) -> None:
        jax.debug.callback(lambda args, kwargs: function(*args, **kwargs), args, kwargs)

    return wrapper
