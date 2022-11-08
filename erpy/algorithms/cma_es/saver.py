from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TYPE_CHECKING, List

from erpy.algorithms.cma_es.population import CMAESPopulation
from erpy.base.genome import ESGenome
from erpy.base.population import Population
from erpy.base.saver import Saver, SaverConfig

if TYPE_CHECKING:
    from erpy.base.ea import EAConfig


@dataclass
class CMAESSaverConfig(SaverConfig):
    @property
    def saver(self) -> Type[CMAESSaver]:
        return CMAESSaver


class CMAESSaver(Saver):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(config)

    def save(self, population: CMAESPopulation) -> None:
        if self.should_save(population.generation):
            # Save the optimizer instance
            path = Path(self.config.save_path) / f"optimizer.pickle"
            with open(path, 'wb') as handle:
                pickle.dump(population.optimizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Save the best genome
            path = Path(self.config.save_path) / f"elite.pickle"
            population.best_genome.save(path)

    def load(self) -> List[ESGenome]:
        elite_path = Path(self.config.save_path) / "elite.pickle"

        with open(elite_path, 'rb') as handle:
            elite = pickle.load(handle)

        return [elite]

    def load_checkpoint(self, checkpoint_path: str, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> CMAESSaverConfig:
        return self._config
