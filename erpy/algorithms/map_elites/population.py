from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Dict, Callable, Optional

from erpy.algorithms.map_elites.map_elites import MAPElitesCell
from erpy.algorithms.map_elites.types import PhenomeDescriptor
from erpy.base.evaluator import EvaluationResult
from erpy.base.genomes import Genome
from erpy.base.population import PopulationConfig, Population


@dataclass
class MAPElitesPopulationConfig(PopulationConfig):
    archive_dimensions: Tuple[int, ...]
    to_phenotypic_descriptor: Callable[
        [[MAPElitesPopulationConfig], Optional[Genome]], PhenomeDescriptor]
    morphological_innovation_protection: bool


class MAPElitesPopulation(Population):
    def __init__(self, config: MAPElitesPopulationConfig) -> None:
        super(MAPElitesPopulation, self).__init__(config=config)

        self._archive: Dict[PhenomeDescriptor, MAPElitesCell] = dict()
        self._archive_times_selected: Dict[PhenomeDescriptor, int] = defaultdict(int)

    @property
    def config(self) -> MAPElitesPopulationConfig:
        return self.config

    def _add_to_archive(self, evaluation_result: EvaluationResult) -> None:
        genome = self.genomes[evaluation_result.genome_id]
        descriptor = self.config.to_phenotypic_descriptor(self.config, genome)

        if descriptor in self._archive:
            cell = self._archive[descriptor]
            # Cell is already occupied -> check if the new genome is better
            if cell.genome.genome_id == evaluation_result.genome_id:
                # The same genome is the occupant, only replace the evaluation result with the new one
                cell.evaluation_result = evaluation_result
                cell.should_save = True
            else:
                # The occupant is a different genome
                if cell.evaluation_result.fitness <= evaluation_result.fitness:
                    # New genome is better -> replace the old one
                    self._archive[descriptor] = MAPElitesCell(genome=genome, evaluation_result=evaluation_result)
                    self._archive[descriptor].should_save = True
                elif self.config.morphological_innovation_protection and cell.genome.age > genome.age:
                    # Morphological innovation protection:
                    #   increase competition fairness by forcing additional training of the new genome
                    self._genomes[genome.genome_id] = genome
                    self.to_evaluate.append(genome.genome_id)
        else:
            # Cell is still empty -> place the genome in it
            self._archive[descriptor] = MAPElitesCell(genome=genome, evaluation_result=evaluation_result)
            self._archive[descriptor].should_save = True

    @property
    def coverage(self) -> float:
        x_dim, y_dim = self.config.archive_dimensions
        coverage = len(self.archive) / (x_dim * y_dim)
        return coverage

    @property
    def genomes(self) -> Dict[int, Genome]:
        # Update the current _genomes datastructure with the current archive
        archive_genomes = {cell.genome.genome_id: cell.genome for cell in self._archive.values()}
        self._genomes.update(archive_genomes)

        return self._genomes

    @property
    def archive(self) -> Dict[PhenomeDescriptor, MAPElitesCell]:
        return self._archive

    @property
    def archive_times_selected(self) -> Dict[PhenomeDescriptor, int]:
        return self._archive_times_selected

    def before_evaluation(self) -> None:
        pass

    def after_evaluation(self) -> None:
        # cleanup
        self.to_reproduce.clear()
        self.to_evaluate.clear()

        # Sort evaluation results based on total number of timesteps
        #   to avoid placing younger results in the archive before older ones
        self.evaluation_results = sorted(self.evaluation_results, key=lambda er: self.genomes[er.genome_id].age,
                                         reverse=True)
        for evaluation_result in self.evaluation_results:
            self._add_to_archive(evaluation_result=evaluation_result)
        self.evaluation_results.clear()

        # Only maintain genomes in the _genomes structure if they require additional training
        #   (i.e. they are in to_evaluate)
        self._genomes = {genome_id: genome for genome_id, genome in self._genomes.items() if
                         genome_id in self.to_evaluate or genome_id in self.under_evaluation}
