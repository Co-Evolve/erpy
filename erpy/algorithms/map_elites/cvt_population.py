from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Type

import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import MiniBatchKMeans

from erpy.algorithms.map_elites.map_elites_cell import MAPElitesCell
from erpy.algorithms.map_elites.types import CellIndex, PhenomeDescription
from erpy.base.ea import EAConfig
from erpy.base.evaluator import EvaluationResult
from erpy.base.genome import Genome
from erpy.base.population import Population, PopulationConfig


@dataclass
class CVTMAPElitesPopulationConfig(PopulationConfig):
    descriptor_size: int
    num_niches: int
    morphological_innovation_protection: bool

    num_init_samples: int = 1000000

    @property
    def population(self) -> Type[CVTMAPElitesPopulation]:
        return CVTMAPElitesPopulation


class CVTMAPElitesPopulation(Population):
    def __init__(self, config: EAConfig) -> None:
        super(CVTMAPElitesPopulation, self).__init__(config=config)

        self._archive: Dict[CellIndex, MAPElitesCell] = dict()
        self._archive_times_selected: Dict[CellIndex, int] = defaultdict(int)
        self._kdt: KDTree = self._initialise_kdt()

    @property
    def config(self) -> CVTMAPElitesPopulationConfig:
        return super().config

    def _initialise_kdt(self) -> KDTree:
        # Generate random init data
        data = np.random.rand(self.config.num_init_samples, self.config.descriptor_size)

        # Extract centroids
        k_means = MiniBatchKMeans(n_clusters=self.config.num_niches,
                                  random_state=self._ea_config.reproducer_config.genome_config.random_state)
        k_means.fit(data)
        centroids = k_means.cluster_centers_

        # Create KDTree
        return KDTree(data=centroids)

    def get_cell_index(self, phenome_descriptor: PhenomeDescription) -> CellIndex:
        # Cell index is the centroid here
        _, index = self._kdt.query(phenome_descriptor)
        return index

    def _add_to_archive(self, evaluation_result: EvaluationResult) -> None:
        genome = self.genomes[evaluation_result.genome_id]
        descriptor = evaluation_result.info["PhenomeDescriptorCallback"]

        cell_index = self.get_cell_index(phenome_descriptor=descriptor)

        if cell_index in self._archive:
            cell = self._archive[cell_index]
            # Cell is already occupied -> check if the new genome is better
            if cell.genome.genome_id == evaluation_result.genome_id:
                # The same genome is the occupant, only replace the evaluation result with the new one
                cell.evaluation_result = evaluation_result
                cell.should_save = True
            else:
                # The occupant is a different genome
                if cell.evaluation_result.fitness <= evaluation_result.fitness:
                    # New genome is better -> replace the old one
                    self._archive[cell_index] = MAPElitesCell(descriptor=cell_index, genome=genome,
                                                              evaluation_result=evaluation_result)
                    self._archive[cell_index].should_save = True
                elif self.config.morphological_innovation_protection and cell.genome.age > genome.age:
                    # Morphological innovation protection:
                    #   increase competition fairness by forcing additional training of the new genome
                    self._genomes[genome.genome_id] = genome
                    self.to_evaluate.append(genome.genome_id)
        else:
            # Cell is still empty -> place the genome in it
            self._archive[cell_index] = MAPElitesCell(descriptor=descriptor, genome=genome,
                                                      evaluation_result=evaluation_result)
            self._archive[cell_index].should_save = True

    @property
    def coverage(self) -> float:
        coverage = len(self.archive) / self.config.num_niches
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

    @property
    def archive_times_selected(self) -> Dict[CellIndex, int]:
        return self._archive_times_selected

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
