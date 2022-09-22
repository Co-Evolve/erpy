from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Dict

import cma

from erpy.base import genome
from erpy.base.ea import EAConfig
from erpy.base.evaluator import EvaluationResult
from erpy.base.genome import ESGenome
from erpy.base.population import PopulationConfig, Population


@dataclass
class CMAESPopulationConfig(PopulationConfig):
    @property
    def population(self) -> Type[CMAESPopulation]:
        return CMAESPopulation


class CMAESPopulation(Population):
    def __init__(self, config: EAConfig):
        super().__init__(config)

        self.optimizer: cma.CMAEvolutionStrategy = None
        self.best_genome: ESGenome = None
        self.best_er: EvaluationResult = None

    @property
    def genomes(self) -> Dict[int, genome.ESGenome]:
        return super().genomes

    def after_evaluation(self) -> None:
        # Extract parameters
        genomes = [self.genomes[er.genome_id] for er in self.evaluation_results]
        solutions = [genome.parameters for genome in genomes]

        # Fitness is negated -> ERPY always maximizes; CMA-ES minimizes
        scores = [-er.fitness for er in self.evaluation_results]

        # Update best genome
        for er in self.evaluation_results:
            if self.best_genome is None or er.fitness > self.best_er.fitness:
                self.best_genome = self.genomes[er.genome_id]
                self.best_er = er

        self.optimizer.tell(solutions=solutions, function_values=scores)
