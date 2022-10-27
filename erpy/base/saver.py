from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TYPE_CHECKING, List

from erpy.base.genome import Genome
from erpy.base.population import Population
from erpy.base.reproducer import Reproducer

if TYPE_CHECKING:
    from erpy.base.ea import EAConfig


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

    def should_save(self, generation: int) -> bool:
        return generation % self.config.save_freq == 0

    @abc.abstractmethod
    def save(self, population: Population) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def load(self) -> List[Genome]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_checkpoint(self, checkpoint_path: str, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> SaverConfig:
        return self._config
