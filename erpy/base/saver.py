from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Type

from base.population import Population


@dataclass
class SaverConfig:
    save_freq: int
    save_path: str

    @property
    @abc.abstractmethod
    def saver(self) -> Type[Saver]:
        raise NotImplementedError


class Saver(metaclass=abc.ABCMeta):
    def __init__(self, config: SaverConfig) -> None:
        self._config = config

    @abc.abstractmethod
    def save(self, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> SaverConfig:
        return self._config
