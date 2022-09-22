from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Type

from erpy.algorithms.map_elites.population import MAPElitesPopulation
from erpy.base.ea import EAConfig
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
                    cell_path = str(output_path / f"cell_{str(descriptor)}")
                    cell.genome.save(cell_path)
                    cell.should_save = False
