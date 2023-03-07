from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import cma
import numpy as np

from erpy import seed
from erpy.framework.ea import EAConfig
from erpy.framework.parameters import ContinuousParameter
from erpy.framework.reproducer import ReproducerConfig, Reproducer
from erpy.instances.populations.cma_es import CMAESPopulation
from erpy.utils.math import renormalize


@dataclass
class CMAESReproducerConfig(ReproducerConfig):
    x0: np.ndarray
    sigma0: float

    @property
    def reproducer(self) -> Type[CMAESReproducer]:
        return CMAESReproducer


class CMAESReproducer(Reproducer):
    def __init__(self, config: EAConfig):
        super().__init__(config)
        self._parameterizer = self.config.genome_config.specification_parameterizer

    @property
    def config(self) -> CMAESReproducerConfig:
        return super().config

    def initialise_population(self, population: CMAESPopulation) -> None:
        options = cma.CMAOptions()
        options.set('bounds', [0, 1])
        options.set('seed', seed)
        population.optimizer = cma.CMAEvolutionStrategy(x0=self.config.x0, sigma0=self.config.sigma0,
                                                        inopts=options)

    def reproduce(self, population: CMAESPopulation) -> None:
        # Ask solutions to the cma_es instance and turn them into genomes
        new_parameters = population.optimizer.ask(number=population.config.population_size)

        for parameters in new_parameters:
            genome_id = self.next_genome_id

            genome = self.config.genome_config.genome.generate(config=self.config.genome_config,
                                                               genome_id=genome_id)
            target_parameters = self._parameterizer.get_target_parameters(genome.specification)
            for parameter_value, target_parameter in zip(parameters, target_parameters):
                assert isinstance(target_parameter,
                                  ContinuousParameter), "CMA-ES only works with continuous parameters."
                target_parameter.value = renormalize(data=parameter_value,
                                                     original_range=[0, 1],
                                                     target_range=[target_parameter.low, target_parameter.high])

            population.genomes[genome_id] = genome
            population.to_evaluate.add(genome_id)
