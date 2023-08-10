from __future__ import annotations

import abc
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Type

from erpy.framework import genome
from erpy.framework.component import EAComponent, EAComponentConfig
from erpy.framework.population import Population

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class ReproducerConfig(EAComponentConfig):
    genome_config: genome.GenomeConfig

    @property
    @abc.abstractmethod
    def reproducer(
            self
            ) -> Type[Reproducer]:
        raise NotImplementedError


class Reproducer(EAComponent):
    def __init__(
            self,
            config: EAConfig
            ) -> None:
        super().__init__(config)
        self._genome_indexer = count(0)

    @property
    def config(
            self
            ) -> ReproducerConfig:
        return self.ea_config.reproducer_config

    @abc.abstractmethod
    def initialise_population(
            self,
            population: Population
            ) -> None:
        raise NotImplementedError

    def initialise_from_checkpoint(
            self,
            population: Population
            ) -> None:
        key = 'reproducer-genome-indexer'
        try:
            self._genome_indexer = population.saving_data[key]
        except KeyError:
            self._genome_indexer = count(0)
            population.saving_data[key] = self._genome_indexer

    @abc.abstractmethod
    def reproduce(
            self,
            population: Population
            ) -> None:
        raise NotImplementedError

    @property
    def next_genome_id(
            self
            ) -> int:
        return next(self._genome_indexer)
