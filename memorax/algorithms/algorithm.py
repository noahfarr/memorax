from typing import Callable, Protocol

from memorax.utils.typing import Key


class State(Protocol):
    step: int
    ...


class Algorithm(Protocol):
    init: Callable[[Key], tuple[Key, State]]
    warmup: Callable[[Key, State, int], tuple[Key, State]]
    train: Callable[[Key, State, int], tuple[Key, State]]
    evaluate: Callable[[Key, State, int], tuple[Key, State]]
