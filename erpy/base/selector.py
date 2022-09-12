from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Type

from erpy.base.population import Population


@dataclass
class SelectorConfig:
    @property
    @abc.abstractmethod
    def selector(self) -> Type[Selector]:
        raise NotImplementedError


class Selector(metaclass=abc.ABCMeta):
    def __init__(self, config: SelectorConfig) -> None:
        self._config = config

    @abc.abstractmethod
    def select(self, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> SelectorConfig:
        return self._config
