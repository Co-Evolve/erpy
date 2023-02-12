from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import cma
import numpy as np

from erpy.instances.algorithms.cma_es.population import CMAESPopulation
from erpy.framework.ea import EAConfig
from erpy.framework.genome import ESGenomeConfig
from erpy.framework.reproducer import ReproducerConfig, Reproducer


@dataclass
class CMAESReproducerConfig(ReproducerConfig):
    genome_config: Type[ESGenomeConfig]
    x0: np.ndarray
    sigma0: float

    @property
    def reproducer(self) -> Type[CMAESReproducer]:
        return CMAESReproducer


class CMAESReproducer(Reproducer):
    def __init__(self, config: EAConfig):
        super().__init__(config)

    @property
    def config(self) -> CMAESReproducerConfig:
        return super().config

    def initialise_population(self, population: CMAESPopulation) -> None:
        options = cma.CMAOptions()
        options.set('bounds', [0, 1])
        options.set('seed', self._ea_config.evaluator_config.environment_config.seed)
        population.optimizer = cma.CMAEvolutionStrategy(x0=self.config.x0, sigma0=self.config.sigma0,
                                                        inopts=options)

    def reproduce(self, population: CMAESPopulation) -> None:
        population.genomes.clear()

        # Ask solutions to the cma_es instance and turn them into genomes
        new_solutions = population.optimizer.ask(number=population.population_size)

        for solution in new_solutions:
            genome_id = self.next_genome_id

            genome = self.config.genome_config.genome(parameters=solution, config=self.config.genome_config,
                                                        genome_id=genome_id)

            population.genomes[genome_id] = genome
            population.to_evaluate.append(genome_id)

        population.to_reproduce.clear()
