from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Type, Union, Any, Iterable

import numpy as np
import wandb

from erpy.framework.ea import EAConfig
from erpy.framework.logger import Logger, LoggerConfig
from erpy.framework.population import Population
from erpy.utils.config2json import config2dict

WandBRun = wandb.wandb_sdk.wandb_run.Run


@dataclass
class WandBLoggerConfig(LoggerConfig):
    project_name: str
    group: str
    tags: List[str]
    update_saver_path: bool
    pre_initialise_wandb: bool
    enable_tensorboard_backend: bool
    _run_name: str

    @property
    def run_name(self) -> str:
        return self._run_name

    @run_name.setter
    def run_name(self, name: str) -> None:
        self._run_name = name

    @property
    def logger(self) -> Type[WandBLogger]:
        return WandBLogger


def wandb_log_values(run: WandBRun, name: str, values: List[float], step: int) -> None:
    run.log({f'{name}_max': np.max(values),
             f'{name}_min': np.min(values),
             f'{name}_mean': np.mean(values),
             f'{name}_std': np.std(values)}, step=step)


def wandb_log_value(run: WandBRun, name: str, value: Union[float, int], step: int) -> None:
    run.log({name: value}, step=step)


def wandb_log_unknown(run: WandBRun, name: str, data: Any, step: int) -> None:
    if isinstance(data, Iterable):
        wandb_log_values(run=run, name=name, values=data, step=step)
    else:
        wandb_log_value(run=run, name=name, value=data, step=step)


class WandBLogger(Logger):
    def __init__(self, config: EAConfig):
        super().__init__(config=config)

        self.run = None
        if self.config.pre_initialise_wandb:
            self._initialise_wandb()

    def _initialise_wandb(self) -> None:
        self.run = wandb.init(project=self.config.project_name,
                              group=self.config.group,
                              tags=self.config.tags,
                              config=config2dict(self.config),
                              sync_tensorboard=self.config.enable_tensorboard_backend)
        self.config.run_name = self.run.name
        self._update_saver_path()

    @property
    def config(self) -> WandBLoggerConfig:
        return super().config

    def _update_saver_path(self):
        if self.config.update_saver_path:
            # Update the saver's path with wandb's run name
            previous_path = Path(self._ea_config.saver_config.save_path)
            new_path = previous_path / self.run.name
            new_path.mkdir(exist_ok=True, parents=True)
            self._ea_config.saver_config.save_path = str(new_path)

    def _log_fitness(self, population: Population) -> None:
        fitnesses = [er.fitness for er in population.evaluation_results]
        wandb_log_values(run=self.run, name='generation/fitness', values=fitnesses, step=population.generation)

    def _log_population_data(self, population: Population) -> None:
        for name, data in population.logging_data:
            wandb_log_unknown(run=self.run, name=name, data=data, step=population.generation)

    def _log_evaluation_result_data(self, population: Population) -> None:
        # log info from evaluation result's info
        try:
            er_log_keys = [key for key in population.evaluation_results[0].info.keys() if
                           key.startswith('logging_')]
            for key in er_log_keys:
                name = "evaluation_results/" + key.replace("logging_", "")
                values = [er.info[key] for er in population.evaluation_results]
                wandb_log_unknown(run=self.run, name=name, data=values, step=population.generation)
        except IndexError:
            pass

    def _log_failures(self, population: Population) -> None:
        failures = [er.info["episode_failures"] for er in population.evaluation_results]
        physics_failures = sum([er_failure["physics"] for er_failure in failures])
        wandb_log_value(run=self.run, name="episode_failures", value=physics_failures, step=population.generation)

    def log(self, population: Population) -> None:
        if self.run is None:
            self._initialise_wandb()
        self._log_fitness(population)
        self._log_population_data(population)
        self._log_evaluation_result_data(population)
        self._log_failures(population)
