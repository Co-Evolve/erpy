from __future__ import annotations

from dataclasses import dataclass
from typing import Type, TYPE_CHECKING

from erpy.framework.population import Population, PopulationConfig

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class DefaultPopulationConfig(PopulationConfig):
    @property
    def population(self) -> Type[Population]:
        return DefaultPopulation


class DefaultPopulation(Population):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(config)

    @property
    def config(self) -> DefaultPopulationConfig:
        return super().config
