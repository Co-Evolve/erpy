from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import numpy as np

from erpy.framework.ea import EAConfig
from erpy.framework.parameters import ContinuousParameter
from erpy.instances.loggers.wandb_logger import WandBLoggerConfig, WandBLogger, wandb_log_value
from erpy.instances.populations.cma_es import CMAESPopulation
from erpy.utils.math import renormalize


@dataclass
class CMAESLoggerConfig(WandBLoggerConfig):
    @property
    def logger(self) -> Type[CMAESLogger]:
        return CMAESLogger


class CMAESLogger(WandBLogger):
    def __init__(self, config: EAConfig):
        super().__init__(config=config)
        self._parameterizer = self._ea_config.reproducer_config.genome_config.specification_parameterizer

        dummy_spec = self._parameterizer.generate_parameterized_specification()
        self._parameter_labels = self._parameterizer.get_parameter_labels(dummy_spec)
        self._parameters = self._parameterizer.get_target_parameters(dummy_spec)

    @property
    def config(self) -> CMAESLoggerConfig:
        return super().config

    def _log_parameters(self, prefix: str, parameters: np.ndarray, step: int, scale2param: bool = True) -> None:
        for parameter_value, label, parameter in zip(parameters, self._parameter_labels, self._parameters):
            assert isinstance(parameter, ContinuousParameter), "CMA-ES only works with ContinuousParameter."
            wandb_log_value(run=self.run, name=f"CMA-ES/{prefix}/{label}", value=parameter_value, step=step)

    def _log_cma_es_optimizer(self, population: CMAESPopulation) -> None:
        step = population.generation
        best_solution = population.optimizer.result[0]
        distribution_mean = population.optimizer.result[5]
        total_n_evaluations = population.optimizer.result[3]
        standard_deviations = population.optimizer.result[6]

        self._log_parameters(prefix='best', parameters=best_solution, step=step)
        self._log_parameters(prefix='distribution_mean', parameters=distribution_mean, step=step)
        self._log_parameters(prefix='distribution_stds', parameters=standard_deviations, step=step)

        wandb_log_value(run=self.run, name='CMA-ES/total_n_evaluations', value=total_n_evaluations, step=step)

    def log(self, population: CMAESPopulation) -> None:
        super().log(population)
        super()._log_evaluation_result_data(population)
        self._log_cma_es_optimizer(population)
