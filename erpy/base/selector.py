from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Type, TYPE_CHECKING

from erpy.base.population import Population

if TYPE_CHECKING:
    from erpy.base.ea import EAConfig


@dataclass
class SelectorConfig:
    @property
    @abc.abstractmethod
    def selector(self) -> Type[Selector]:
        raise NotImplementedError


class Selector(metaclass=abc.ABCMeta):
    def __init__(self, config: EAConfig) -> None:
        self._ea_config = config
        self._config = config.selector_config

    @abc.abstractmethod
    def select(self, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> SelectorConfig:
        return self._config
