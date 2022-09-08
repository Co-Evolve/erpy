from base.population import Population
from base.reproducer import Reproducer, ReproducerConfig


class MAPElitesReproducer(Reproducer):
    def __init__(self, config: ReproducerConfig) -> None:
        super(MAPElitesReproducer, self).__init__(config)

    def reproduce(self, population: Population) -> None:
        for parent_id in population.to_reproduce:
            parent_genome = population.genomes[parent_id]

            child_genome = parent_genome.mutate(self.next_genome_id)

            # Add the child to the population
            population.genomes[child_genome.genome_id] = child_genome

            # New children should always be evaluated
            population.to_evaluate.append(child_genome.genome_id)

        population.to_reproduce.clear()
