from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import numpy as np

from algorithms.map_elites.population import MAPElitesPopulation
from base.selector import SelectorConfig, Selector


@dataclass
class MAPElitesSelectorConfig(SelectorConfig):
    def selector(self) -> Type[MAPElitesSelector]:
        return MAPElitesSelector


class MAPElitesSelector(Selector):
    def __init__(self, config: MAPElitesSelectorConfig) -> None:
        super(MAPElitesSelector, self).__init__(config)

    def select(self, population: MAPElitesPopulation) -> None:
        # Randomly select genomes
        num_to_select = self.config.population_size - len(population.to_evaluate) - len(population.under_evaluation)

        if num_to_select > 0:
            options = list([descriptor for descriptor, cell in population.archive.items() if
                            cell.genome.genome_id not in population.under_evaluation])
            times_selected = [population.archive_times_selected[cell] for cell in options]
            times_selected = np.argsort(times_selected)

            selected_cells = [options[index] for index in times_selected]

            # todo: select num options
            # selected_cells = list(self.config.random_state.choice(a=options,
            #                                                      size=num_to_select,
            #                                                      replace=False))
            for cell in selected_cells:
                population.archive_times_selected[cell] += 1

            selected_genome_ids = [population.archive[cell].genome.genome_id for cell in selected_cells]
            population.to_reproduce += selected_genome_ids
