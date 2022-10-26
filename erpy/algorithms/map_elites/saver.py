from __future__ import annotations

import glob
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Type, List

from erpy.algorithms.map_elites.map_elites_cell import MAPElitesCell
from erpy.algorithms.map_elites.population import MAPElitesPopulation
from erpy.base.ea import EAConfig
from erpy.base.genome import Genome
from erpy.base.reproducer import Reproducer
from erpy.base.saver import SaverConfig, Saver


@dataclass
class MAPElitesSaverConfig(SaverConfig):
    @property
    def saver(self) -> Type[MAPElitesSaver]:
        return MAPElitesSaver


class MAPElitesSaver(Saver):
    def __init__(self, config: EAConfig):
        super(MAPElitesSaver, self).__init__(config=config)

        self.output_path = Path(self.config.save_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def save(self, population: MAPElitesPopulation) -> None:
        if self.should_save(population.generation):
            # Save the archive
            output_path = self.output_path / "archive"
            output_path.mkdir(parents=True, exist_ok=True)

            for descriptor, cell in population.archive.items():
                if cell.should_save:
                    cell_path = str(output_path / f"cell_{str(descriptor)}.pkl")
                    cell.should_save = False
                    cell.save(cell_path)

    def load(self) -> List[Genome]:
        pass

    def load_checkpoint(self, checkpoint_path: str, population: MAPElitesPopulation,
                        reproducer: Reproducer) -> None:
        # Todo: Remove the reproducer from this call once we have reran the experiment,
        #   instead, save the different components during checkpoint
        path = Path(checkpoint_path)
        assert path.exists(), f"Given checkpoint path does not exist: {checkpoint_path}"
        cell_paths = glob.glob(str(path / 'cell*'))

        max_genome_id = -1
        for cell_path in cell_paths:
            cell = MAPElitesCell.load(cell_path)
            if isinstance(cell, Genome):
                genome_id = cell.genome_id

                # Add genome to the population
                population.genomes[genome_id] = cell

                # Re-evaluate genome
                population.to_evaluate.append(genome_id)

            elif isinstance(cell, MAPElitesCell):
                population.archive[cell.descriptor] = cell
                genome_id = cell.genome.genome_id
            else:
                raise NotImplementedError

            max_genome_id = max(max_genome_id, genome_id)

        reproducer._genome_indexer = count(max_genome_id + 1)
