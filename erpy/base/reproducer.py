from __future__ import annotations

import abc
from dataclasses import dataclass
from itertools import count
from typing import Type, TYPE_CHECKING

from erpy.base import genome
from erpy.base.population import Population

if TYPE_CHECKING:
    from erpy.base.ea import EAConfig


@dataclass
class ReproducerConfig:
    genome_config: genome.GenomeConfig

    @property
    @abc.abstractmethod
    def reproducer(self) -> Type[Reproducer]:
        raise NotImplementedError


class Reproducer(metaclass=abc.ABCMeta):
    def __init__(self, config: EAConfig) -> None:
        self._ea_config = config
        self._config = config.reproducer_config
        self._genome_indexer = count(0)

    @abc.abstractmethod
    def initialise_population(self, population: Population) -> None:
        raise NotImplementedError

    def initialise_from_checkpoint(self, population: Population) -> None:
        key = 'reproducer-genome-indexer'
        try:
            self._genome_indexer = population.saving_data[key]
        except KeyError:
            self._genome_indexer = count(0)
            population.saving_data[key] = self._genome_indexer

    @abc.abstractmethod
    def reproduce(self, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> ReproducerConfig:
        return self._config

    @property
    def next_genome_id(self) -> int:
        return next(self._genome_indexer)
