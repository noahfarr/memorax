import atexit
from typing import Protocol, runtime_checkable

from memorax.utils.typing import PyTree


@runtime_checkable
class Logger(Protocol):
    def log(self, data: PyTree, step: int, **kwargs) -> None: ...
    def finish(self) -> None: ...


class MultiLogger:
    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers
        atexit.register(self.finish)

    def log(self, data: PyTree, step: int, **kwargs) -> None:
        for logger in self.loggers:
            logger.log(data, step, **kwargs)

    def finish(self) -> None:
        for logger in self.loggers:
            logger.finish()
