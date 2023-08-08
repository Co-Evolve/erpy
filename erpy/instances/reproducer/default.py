from __future__ import annotations

from typing import Type

import erpy
from erpy.framework.population import Population
from erpy.framework.reproducer import ReproducerConfig, Reproducer
from erpy.instances.population.default import DefaultPopulation


class DefaultReproducerConfig(ReproducerConfig):
    @property
    def reproducer(self) -> Type[DefaultReproducer]:
        return DefaultReproducer


class DefaultReproducer(Reproducer):
    def __init__(self, config: ReproducerConfig) -> None:
        super().__init__(config)

    def initialise_population(self, population: DefaultPopulation) -> None:
        num_to_generate = population.config.population_size

        for i in range(num_to_generate):
            # Create genome
            genome_id = self.next_genome_id
            genome = self.config.genome_config.genome.generate(config=self.config.genome_config,
                                                               genome_id=genome_id)

            # Add genome to population
            population.genomes[genome_id] = genome

            # Initial genomes should always be evaluated
            population.to_evaluate.add(genome_id)

    def reproduce(self, population: Population) -> None:
        amount_to_create = population.config.population_size - len(population.to_evaluate)

        for _ in range(amount_to_create):
            parent_id = erpy.random_state.choice(list(population.to_reproduce))
            parent_genome = population.genomes[parent_id]
            child_genome = parent_genome.mutate(self.next_genome_id)

            # Add the child to the population
            population.genomes[child_genome.genome_id] = child_genome

            # New children should always be evaluated
            population.to_evaluate.add(child_genome.genome_id)
