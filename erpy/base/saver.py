from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Type, TYPE_CHECKING

from erpy.base.population import Population

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


class Saver(metaclass=abc.ABCMeta):
    def __init__(self, config: EAConfig) -> None:
        self._ea_config = config
        self._config = config.saver_config

    @abc.abstractmethod
    def save(self, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> SaverConfig:
        return self._config
