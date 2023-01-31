from __future__ import annotations

from typing import Type

from erpy.framework.population import Population
from erpy.framework.reproducer import ReproducerConfig, Reproducer


class DefaultReproducerConfig(ReproducerConfig):
    @property
    def reproducer(self) -> Type[DefaultReproducer]:
        return DefaultReproducer


class DefaultReproducer(Reproducer):
    def __init__(self, config: ReproducerConfig) -> None:
        super().__init__(config)

    def initialise_population(self, population: Population) -> None:
        num_to_generate = population.population_size

        for i in range(num_to_generate):
            # Create genome
            genome_id = self.next_genome_id
            genome = self.config.genome_config.genome.generate(config=self.config.genome_config,
                                                               genome_id=genome_id)

            # Add genome to population
            population.genomes[genome_id] = genome

            # Initial genomes should always be evaluated
            population.to_evaluate.append(genome_id)

    def reproduce(self, population: Population) -> None:
        for parent_id in population.to_reproduce:
            parent_genome = population.genomes[parent_id]

            child_genome = parent_genome.mutate(self.next_genome_id)

            # Add the child to the population
            population.genomes[child_genome.genome_id] = child_genome

            # New children should always be evaluated
            population.to_evaluate.append(child_genome.genome_id)

        population.to_reproduce.clear()
