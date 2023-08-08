from erpy.framework.genome import GenomeConfig, Genome
from erpy.framework.specification import RobotSpecification


class DummyGenome(Genome):
    def __init__(self, genome_id: int, specification: RobotSpecification) -> None:
        super(DummyGenome, self).__init__(config=None, genome_id=genome_id, parent_genome_id=None)
        self._specification = specification

    @property
    def specification(self) -> RobotSpecification:
        return self._specification

    @staticmethod
    def generate(config: GenomeConfig, genome_id: int, *args, **kwargs) -> Genome:
        raise NotImplementedError

    def mutate(self, child_genome_id: int, *args, **kwargs) -> Genome:
        raise NotImplementedError

    def cross_over(self, partner_genome: Genome, child_genome_id: int) -> Genome:
        raise NotImplementedError
