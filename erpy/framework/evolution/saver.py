from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Type

from erpy.framework.component import EAComponent, EAComponentConfig
from erpy.framework.population import Population

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class SaverConfig(EAComponentConfig):
    save_freq: int
    save_path: str

    @property
    @abc.abstractmethod
    def saver(
            self
            ) -> Type[Saver]:
        raise NotImplementedError

    @property
    def analysis_path(
            self
            ) -> str:
        return str(Path(self.save_path) / "analysis")


class Saver(EAComponent):
    def __init__(
            self,
            config: EAConfig
            ) -> None:
        super().__init__(config)

        Path(self.config.save_path).mkdir(parents=True, exist_ok=True)

    @property
    def config(
            self
            ) -> SaverConfig:
        return self.ea_config.saver_config

    def should_save(
            self,
            generation: int
            ) -> bool:
        return generation % self.config.save_freq == 0 or generation == self._ea_config.num_generations

    @abc.abstractmethod
    def save(
            self,
            population: Population
            ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def load(
            self
            ) -> Population:
        raise NotImplementedError
