from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Type, TYPE_CHECKING

from erpy.framework.population import Population

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig

@dataclass
class LoggerConfig:
    @property
    @abc.abstractmethod
    def logger(self) -> Type[Logger]:
        raise NotImplementedError


class Logger(metaclass=abc.ABCMeta):
    def __init__(self, config: EAConfig) -> None:
        self._ea_config = config
        self._config = config.logger_config

    @abc.abstractmethod
    def log(self, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> LoggerConfig:
        return self._config
