from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Callable, Set

from erpy.base.genome import Genome
from erpy.base.population import Population
from erpy.base.reproducer import ReproducerConfig, Reproducer


@dataclass
class UniqueReproducerConfig(ReproducerConfig):
    uniqueness_test: Callable[[Set, Genome], bool]

    @property
    def reproducer(self) -> Type[UniqueReproducer]:
        return UniqueReproducer


class UniqueReproducer(Reproducer):
    def __init__(self, config: UniqueReproducerConfig) -> None:
        super().__init__(config=config)
        self._archive = set()

    @property
    def config(self) -> UniqueReproducerConfig:
        return super().config

    def initialise_population(self, population: Population) -> None:
        num_to_generate = population.population_size

        for i in range(num_to_generate):
            # Create genome
            genome_id = self.next_genome_id
            genome = self.config.genome_config.genome.generate(config=self.config.genome_config,
                                                               genome_id=genome_id)

            while not self.config.uniqueness_test(self._archive, genome):
                genome = self.config.genome_config.genome.generate(config=self.config.genome_config,
                                                                   genome_id=genome_id)

            # Add genome to population
            population.genomes[genome_id] = genome

            # Initial genomes should always be evaluated
            population.to_evaluate.append(genome_id)

    def reproduce(self, population: Population) -> None:
        for parent_id in population.to_reproduce:
            parent_genome = population.genomes[parent_id]

            child_id = self.next_genome_id
            child_genome = parent_genome.mutate(child_id)

            while not self.config.uniqueness_test(self._archive, child_genome):
                child_id = self.next_genome_id
                child_genome = parent_genome.mutate(child_id)

            # Add the child to the population
            population.genomes[child_genome.genome_id] = child_genome

            # New children should always be evaluated
            population.to_evaluate.append(child_genome.genome_id)

        population.to_reproduce.clear()
