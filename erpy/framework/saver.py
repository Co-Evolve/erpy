from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TYPE_CHECKING

from erpy.framework.population import Population

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class SaverConfig:
    save_freq: int
    save_path: str

    @property
    @abc.abstractmethod
    def saver(self) -> Type[Saver]:
        raise NotImplementedError

    @property
    def analysis_path(self) -> str:
        return str(Path(self.save_path) / "analysis")


class Saver(metaclass=abc.ABCMeta):
    def __init__(self, config: EAConfig) -> None:
        self._ea_config = config
        self._config = config.saver_config

        Path(self.config.save_path).mkdir(parents=True, exist_ok=True)

    def should_save(self, generation: int) -> bool:
        return generation % self.config.save_freq == 0 or generation == self._ea_config.num_generations

    @abc.abstractmethod
    def save(self, population: Population) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def load(self) -> Population:
        raise NotImplementedError

    @property
    def config(self) -> SaverConfig:
        return self._config
