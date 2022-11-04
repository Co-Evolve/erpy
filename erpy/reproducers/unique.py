from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Callable, Set

from erpy.base.genome import Genome
from erpy.base.population import Population
from erpy.base.reproducer import ReproducerConfig, Reproducer


@dataclass
class UniqueReproducerConfig(ReproducerConfig):
    uniqueness_test: Callable[[Set, Genome, Population], bool]
    max_retries: int

    @property
    def reproducer(self) -> Type[UniqueReproducer]:
        return UniqueReproducer


class UniqueReproducer(Reproducer):
    def __init__(self, config: UniqueReproducerConfig) -> None:
        super().__init__(config=config)
        self._archive = None

    @property
    def config(self) -> UniqueReproducerConfig:
        return super().config

    def _initialise_from_checkpoint(self, population: Population) -> None:
        super()._initialise_from_checkpoint(population=population)

        key = "unique-reproducer-archive"
        try:
            self._archive = population.saving_data[key]
        except KeyError:
            self._archive = set()
            population.saving_data[key] = self._archive

    def initialise_population(self, population: Population) -> None:
        self._initialise_from_checkpoint(population)

        num_to_generate = population.population_size

        for i in range(num_to_generate):
            # Create genome
            genome_id = self.next_genome_id
            genome = self.config.genome_config.genome.generate(config=self.config.genome_config,
                                                               genome_id=genome_id)

            num_retries = 0
            while not self.config.uniqueness_test(self._archive, genome,
                                                  population) and num_retries < self.config.max_retries:
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

            num_retries = 0
            while not self.config.uniqueness_test(self._archive, child_genome,
                                                  population) and num_retries < self.config.max_retries:
                # Continue mutating the same genome until it is unique
                child_genome.genome_id = parent_id
                child_genome = child_genome.mutate(child_id)

            # Add the child to the population
            population.genomes[child_genome.genome_id] = child_genome

            # New children should always be evaluated
            population.to_evaluate.append(child_genome.genome_id)

        population.to_reproduce.clear()

        population.logging_data["UniqueReproducer/number_of_unique_genomes"] = len(self._archive)
