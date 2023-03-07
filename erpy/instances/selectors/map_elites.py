from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Type

import numpy as np

from erpy.framework.ea import EAConfig
from erpy.framework.selector import SelectorConfig, Selector
from erpy.instances.populations.map_elites import MAPElitesPopulation


@dataclass
class MAPElitesSelectorConfig(SelectorConfig):
    @property
    def selector(self) -> Type[MAPElitesSelector]:
        return MAPElitesSelector


class MAPElitesSelector(Selector):
    def __init__(self, config: EAConfig) -> None:
        super(MAPElitesSelector, self).__init__(config=config)

        self._number_of_selections_per_cell = defaultdict(int)

    def select(self, population: MAPElitesPopulation) -> None:
        num_to_select = population.config.population_size - len(population.to_evaluate) - len(
            population.under_evaluation)
        options = list([cell_index for cell_index, cell in population.archive.items() if
                        cell.genome.genome_id not in population.under_evaluation])

        if num_to_select > 0 and len(options) > 0:
            times_selected = [self._number_of_selections_per_cell[cell_index] for cell_index in options]
            times_selected = np.argsort(times_selected)

            selected_cell_indices = [options[index] for index in times_selected[:num_to_select]]

            for cell_index in selected_cell_indices:
                self._number_of_selections_per_cell[cell_index] += 1

            selected_genome_ids = [population.archive[cell_index].genome.genome_id for cell_index in
                                   selected_cell_indices]
            for selected_genome_id in selected_genome_ids:
                population.to_reproduce.add(selected_genome_id)
