from pathlib import Path

from erpy.algorithms.map_elites.population import MAPElitesPopulation
from erpy.base.saver import Saver, SaverConfig


class MAPElitesSaver(Saver):
    def __init__(self, config: SaverConfig):
        super(MAPElitesSaver, self).__init__(config=config)

        self.output_path = Path(self.config.save_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def save(self, population: MAPElitesPopulation) -> None:
        if population.generation % self.config.save_freq == 0:
            # Save the archive
            output_path = self.output_path / "archive"
            output_path.mkdir(parents=True, exist_ok=True)

            for descriptor, cell in population.archive.items():
                if cell.should_save:
                    cell_path = str(output_path / f"cell_{str(descriptor)}")
                    cell.genome.save(cell_path)
                    cell.should_save = False
