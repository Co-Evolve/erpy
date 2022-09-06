import abc
from dataclasses import dataclass

from erpy.base.population import Population


@dataclass
class SelectorConfig:
    population_size: int


class Selector(metaclass=abc.ABCMeta):
    def __init__(self, config: SelectorConfig) -> None:
        self._config = config

    @abc.abstractmethod
    def select(self, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> SelectorConfig:
        return self._config
