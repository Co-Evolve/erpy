from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import cma

from erpy.framework.ea import EAConfig
from erpy.framework.parameters import ContinuousParameter
from erpy.framework.population import PopulationConfig, Population
from erpy.utils.math import renormalize


@dataclass
class CMAESPopulationConfig(PopulationConfig):
    @property
    def population(self) -> Type[CMAESPopulation]:
        return CMAESPopulation


class CMAESPopulation(Population):
    def __init__(self, config: EAConfig):
        super().__init__(config)

        self.optimizer: cma.CMAEvolutionStrategy = None
        self._parameterizer = self._ea_config.reproducer_config.genome_config.specification_parameterizer

    def after_evaluation(self) -> None:
        super().after_evaluation()

        # Extract parameters
        genomes = [er.genome for er in self.evaluation_results]
        all_parameters = []
        for genome in genomes:
            genome_parameters = []
            target_parameters = self._parameterizer.get_target_parameters(genome.specification)
            for target_parameter in target_parameters:
                assert isinstance(target_parameter,
                                  ContinuousParameter), "CMA-ES only works with continuous parameters."
                value = renormalize(data=target_parameter.value,
                                    original_range=[target_parameter.low, target_parameter.high],
                                    target_range=[0, 1])
                genome_parameters.append(value)
            all_parameters.append(genome_parameters)

        # Fitness is negated -> ERPY always maximizes; CMA-ES minimizes
        scores = [-er.fitness for er in self.evaluation_results]

        self.optimizer.tell(solutions=all_parameters, function_values=scores)
