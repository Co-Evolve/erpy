import abc
from dataclasses import dataclass

from erpy.base.population import Population


@dataclass
class LoggerConfig:
    pass


class Logger(metaclass=abc.ABCMeta):
    def __init__(self, config: LoggerConfig) -> None:
        self._config = config

    @abc.abstractmethod
    def log(self, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> LoggerConfig:
        return self._config
