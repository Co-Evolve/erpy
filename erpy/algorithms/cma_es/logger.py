from __future__ import annotations

from dataclasses import dataclass
from typing import Type, List, Optional

import numpy as np

from erpy.algorithms.cma_es.population import CMAESPopulation
from erpy.base.ea import EAConfig
from erpy.base.population import Population
from erpy.loggers.wandb_logger import WandBLogger, WandBLoggerConfig


@dataclass
class CMAESLoggerConfig(WandBLoggerConfig):
    parameter_labels: Optional[List[str]]

    @property
    def logger(self) -> Type[CMAESLogger]:
        return CMAESLogger


class CMAESLogger(WandBLogger):
    def __init__(self, config: EAConfig):
        super().__init__(config=config)

    @property
    def config(self) -> CMAESLoggerConfig:
        return super().config

    def _log_parameters(self, prefix: str, parameters: np.ndarray, step: int) -> None:
        if self.config.parameter_labels is None:
            self.config.parameter_labels = [f"param_{i}" for i in range(len(parameters))]

        for name, value in zip(self.config.parameter_labels, parameters):
            self._log_value(name=f'CMA-ES/{prefix}/{name}', value=value, step=step)

    def _log_evaluation_result_data(self, population: Population) -> None:
        er_log_keys = [key for key in population.evaluation_results[0].info.keys() if key.startswith('logging_')]
        for key in er_log_keys:
            name = "evaluation_results/" + key.replace("logging_", "")
            values = [er.info[key] for er in population.evaluation_results]
            self._log_unknown(name=name, data=values, step=population.generation)

    def log(self, population: CMAESPopulation) -> None:
        super().log(population)

        step = population.generation
        best_solution = population.optimizer.result[0]
        distribution_mean = population.optimizer.result[5]
        total_n_evaluations = population.optimizer.result[3]
        standard_deviations = population.optimizer.result[6]

        self._log_parameters(prefix='best', parameters=best_solution, step=step)
        self._log_parameters(prefix='distribution_mean', parameters=distribution_mean, step=step)
        self._log_parameters(prefix='distribution_stds', parameters=standard_deviations, step=step)

        self._log_value(name='CMA-ES/total_n_evaluations', value=total_n_evaluations, step=step)
        self._log_value(name='CMA-ES/total_n_evaluations', value=total_n_evaluations, step=step)
