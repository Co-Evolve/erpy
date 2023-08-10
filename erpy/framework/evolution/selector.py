from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

from erpy.framework.component import EAComponent, EAComponentConfig
from erpy.framework.population import Population

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class SelectorConfig(EAComponentConfig):
    @property
    @abc.abstractmethod
    def selector(
            self
            ) -> Type[Selector]:
        raise NotImplementedError


class Selector(EAComponent):
    def __init__(
            self,
            config: EAConfig
            ) -> None:
        super().__init__(config)

    @property
    def config(
            self
            ) -> SelectorConfig:
        return self.ea_config.selector_config

    @abc.abstractmethod
    def select(
            self,
            population: Population
            ) -> None:
        raise NotImplementedError
