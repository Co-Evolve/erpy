from __future__ import annotations

import glob
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TYPE_CHECKING

import numpy as np

from erpy.framework.population import Population
from erpy.framework.saver import SaverConfig, Saver

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class DefaultSaverConfig(SaverConfig):
    @property
    def saver(self) -> Type[Saver]:
        return DefaultSaver


class DefaultSaver(Saver):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(config)

    def save(self, population: Population) -> None:
        if self.should_save(generation=population.generation):
            path = str(Path(self.config.save_path) / f"generation_{population.generation}.pkl")
            with open(path, "wb") as f:
                pickle.dump(obj=population, file=f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> Population:
        path = str(Path(self.config.save_path) / f"generation_*.pkl")
        files = glob.glob(path)
        generations = [int(file.split('_')[-1].split('.')[0]) for file in files]
        highest_generation_index = np.argsort(generations)[-1]
        file = files[highest_generation_index]

        with open(file, "rb") as f:
            population = pickle.load(f)
        return population

    @property
    def config(self) -> SaverConfig:
        return self._config
