from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Type, TYPE_CHECKING

from erpy.base.population import Population

if TYPE_CHECKING:
    from erpy.base.ea import EAConfig


@dataclass
class DummySelectorConfig:
    @property
    def selector(self) -> Type[DummySelector]:
        return DummySelector


class DummySelector(metaclass=abc.ABCMeta):
    def __init__(self, config: EAConfig) -> None:
        self._ea_config = config
        self._config = config.selector_config

    def select(self, population: Population) -> None:
        pass

    @property
    def config(self) -> DummySelectorConfig:
        return self._config
