from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type, List

import numpy as np

from erpy.framework.ea import EAConfig
from erpy.framework.evaluator import EvaluationResult
from erpy.framework.genome import Genome
from erpy.framework.population import Population, PopulationConfig
from erpy.instances.algorithms.map_elites.map_elites_cell import MAPElitesCell
from erpy.instances.algorithms.map_elites.types import CellIndex, PhenomeDescription


@dataclass
class MAPElitesPopulationConfig(PopulationConfig):
    archive_dimensions: List[int]
    morphological_innovation_protection: bool

    @property
    def population(self) -> Type[MAPElitesPopulation]:
        return MAPElitesPopulation


class MAPElitesPopulation(Population):
    def __init__(self, config: EAConfig) -> None:
        super(MAPElitesPopulation, self).__init__(config=config)

        self._archive: Dict[CellIndex, MAPElitesCell] = dict()

    @property
    def config(self) -> MAPElitesPopulationConfig:
        return super().config

    def get_cell_index(self, phenome_descriptor: PhenomeDescription) -> CellIndex:
        archive_dimensions = np.asarray(self.config.archive_dimensions)
        cell_index = tuple(np.rint(phenome_descriptor * (archive_dimensions - 1)).astype(int))

        return cell_index

    @property
    def get_elite(self) -> MAPElitesCell:
        cells = list(self.archive.values())
        fitnesses = [cell.evaluation_result.fitness for cell in cells]
        index = np.argmax(fitnesses)
        return cells[index]

    def _add_to_archive(self, evaluation_result: EvaluationResult) -> None:
        assert "PhenomeDescriptor" in evaluation_result.info, "MAP-Elites requires a PhenomeDescriptor" \
                                                                      "Callback to add a descriptor to the " \
                                                                      "evaluation result."
        genome = self.genomes[evaluation_result.genome.genome_id]
        descriptor = evaluation_result.info["PhenomeDescriptor"]

        cell_index = self.get_cell_index(phenome_descriptor=descriptor)

        if cell_index in self._archive:
            cell = self._archive[cell_index]
            # Cell is already occupied -> check if the new genome is better
            if cell.genome.genome_id == evaluation_result.genome.genome_id:
                # The same genome is the occupant, only replace the evaluation result with the new one
                cell.evaluation_result = evaluation_result
                cell.should_save = True
            else:
                # The occupant is a different genome
                if cell.evaluation_result.fitness <= evaluation_result.fitness:
                    # New genome is better -> replace the old one
                    self._archive[cell_index] = MAPElitesCell(descriptor=cell_index,
                                                              evaluation_result=evaluation_result)
                    self._archive[cell_index].should_save = True
                elif self.config.morphological_innovation_protection and cell.genome.age > genome.age:
                    # Morphological innovation protection:
                    #   increase competition fairness by forcing additional training of the new genome
                    self._genomes[genome.genome_id] = genome
                    self.to_evaluate.add(genome.genome_id)
        else:
            # Cell is still empty -> place the genome in it
            self._archive[cell_index] = MAPElitesCell(descriptor=cell_index,
                                                      evaluation_result=evaluation_result)
            self._archive[cell_index].should_save = True

    @property
    def coverage(self) -> float:
        coverage = len(self.archive) / np.prod(self.config.archive_dimensions)
        return coverage

    @property
    def genomes(self) -> Dict[int, Genome]:
        # Update the current _genomes datastructure with the current archive
        archive_genomes = {cell.genome.genome_id: cell.genome for cell in self._archive.values()}
        self._genomes.update(archive_genomes)

        return self._genomes

    @property
    def archive(self) -> Dict[CellIndex, MAPElitesCell]:
        return self._archive

    def after_evaluation(self) -> None:
        super().after_evaluation()

        # Sort evaluation results based on total number of timesteps
        #   to avoid placing younger results in the archive before older ones
        self.evaluation_results = sorted(self.evaluation_results, key=lambda er: self.genomes[er.genome.genome_id].age,
                                         reverse=True)
        for evaluation_result in self.evaluation_results:
            self._add_to_archive(evaluation_result=evaluation_result)
