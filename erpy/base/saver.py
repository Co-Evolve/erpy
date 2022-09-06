import abc
from dataclasses import dataclass

from erpy.base.population import Population


@dataclass
class SaverConfig:
    save_freq: int
    save_path: str


class Saver(metaclass=abc.ABCMeta):
    def __init__(self, config: SaverConfig) -> None:
        self._config = config

    @abc.abstractmethod
    def save(self, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> SaverConfig:
        return self._config
