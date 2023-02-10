from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Type, TYPE_CHECKING, List

import numpy as np

from erpy.framework.logger import Logger, LoggerConfig
from erpy.framework.population import Population

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class DefaultLoggerConfig(LoggerConfig):
    @property
    def logger(self) -> Type[Logger]:
        return DefaultLogger


class DefaultLogger(Logger):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(config)
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger()
        log.setLevel(logging.INFO)

    @property
    def config(self) -> DefaultLoggerConfig:
        return self._config

    @staticmethod
    def _values_log_string(name: str, values: List[float], generation: int) -> str:
        log_str = f"[Generation {generation}] {name}" \
                  f"\n\tmean: {np.mean(values)}" \
                  f"\n\tmax: {np.max(values)}" \
                  f"\n\tmin: {np.min(values)}"
        return log_str

    def _log_fitness(self, population: Population) -> None:
        fitnesses = [er.fitness for er in population.evaluation_results]
        log_str = self._values_log_string(name="fitness", values=fitnesses, generation=population.generation)
        log_str += f"\n\tall time highest: {population.all_time_best_evaluation_result.fitness}"
        logging.info(log_str)

    def log(self, population: Population) -> None:
        self._log_fitness(population)
