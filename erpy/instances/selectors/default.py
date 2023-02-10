from __future__ import annotations

from dataclasses import dataclass
from typing import Type, TYPE_CHECKING

from erpy.framework.population import Population
from erpy.framework.selector import SelectorConfig, Selector

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class DefaultSelectorConfig(SelectorConfig):
    amount_to_select: int

    @property
    def selector(self) -> Type[Selector]:
        return DefaultSelector


class DefaultSelector(Selector):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(config)

    @property
    def config(self) -> DefaultSelectorConfig:
        return self._config

    def select(self, population: Population) -> None:
        evaluation_results = population.evaluation_results
        sorted_evaluation_results = sorted(evaluation_results, key=lambda er: er.fitness, reverse=True)

        for evaluation_result in sorted_evaluation_results[:self.config.amount_to_select]:
            population.to_reproduce.add(evaluation_result.genome.genome_id)
