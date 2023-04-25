from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TYPE_CHECKING, List

from erpy.framework.genome import Genome
from erpy.framework.saver import SaverConfig, Saver
from erpy.instances.populations.cma_es import CMAESPopulation

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class CMAESSaverConfig(SaverConfig):
    save_optimizer: bool = True

    @property
    def saver(self) -> Type[CMAESSaver]:
        return CMAESSaver


class CMAESSaver(Saver):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(config)

    def save(self, population: CMAESPopulation) -> None:
        if self.should_save(population.generation):
            # Save the populationoptimizer instance
            if self.config.save_optimizer:
                path = Path(self.config.save_path) / f"optimizer.pickle"
                with open(path, 'wb') as handle:
                    pickle.dump(population.optimizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Save the best genome
            path = Path(self.config.save_path) / f"elite.pickle"
            population.all_time_best_evaluation_result.genome.save(path)

    def load(self) -> List[Genome]:
        elite_path = Path(self.config.save_path) / "elite.pickle"

        with open(elite_path, 'rb') as handle:
            elite = pickle.load(handle)

        output = [elite]

        if self.config.save_optimizer:
            optimizer_path = Path(self.config.save_path) / f"optimizer.pickle"
            with open(optimizer_path, 'rb') as handle:
                optimizer = pickle.load(handle)
            output += [optimizer]

        return output

    def load_checkpoint(self, checkpoint_path: str, population: CMAESPopulation) -> None:
        raise NotImplementedError

    @property
    def config(self) -> CMAESSaverConfig:
        return self._config
