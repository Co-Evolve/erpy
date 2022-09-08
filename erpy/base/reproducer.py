import abc
from dataclasses import dataclass
from itertools import count

from base.population import Population


@dataclass
class ReproducerConfig:
    pass

class Reproducer(metaclass=abc.ABCMeta):
    def __init__(self, config: ReproducerConfig) -> None:
        self._config = config
        self._genome_indexer = count(0)

    @abc.abstractmethod
    def reproduce(self, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> ReproducerConfig:
        return self._config

    @property
    def next_genome_id(self) -> int:
        return next(self._genome_indexer)