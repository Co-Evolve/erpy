from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Type, Union, Any, Iterable

import numpy as np
import wandb

from erpy.base.ea import EAConfig
from erpy.base.logger import Logger, LoggerConfig
from erpy.base.population import Population
from erpy.utils.config2json import config2dict


@dataclass
class WandBLoggerConfig(LoggerConfig):
    project_name: str
    group: str
    tags: List[str]
    update_saver_path: bool

    @property
    def logger(self) -> Type[WandBLogger]:
        return WandBLogger


class WandBLogger(Logger):
    def __init__(self, config: EAConfig):
        super(WandBLogger, self).__init__(config=config)

        self.wandb = wandb.init(project=self.config.project_name,
                                group=self.config.group,
                                tags=self.config.tags,
                                config=config2dict(self._ea_config))

        self._update_saver_path()

    @property
    def config(self) -> WandBLoggerConfig:
        return super().config

    def _update_saver_path(self):
        if self.config.update_saver_path:
            # Update the saver's path with wandb's run name
            previous_path = Path(self._ea_config.saver_config.save_path)
            new_path = previous_path / wandb.run.name
            new_path.mkdir(exist_ok=True, parents=True)
            self._ea_config.saver_config.save_path = str(new_path)

    def _log_values(self, name: str, values: List[float], step: int) -> None:
        self.wandb.log({f'{name}_max': np.max(values),
                        f'{name}_mean': np.mean(values),
                        f'{name}_std': np.std(values)}, step=step)

    def _log_value(self, name: str, value: Union[float, int], step: int) -> None:
        self.wandb.log({{name}: value}, step=step)

    def _log_unknown(self, name: str, data: Any, step: int) -> None:
        if isinstance(data, Iterable):
            self._log_values(name=name, values=data, step=step)
        else:
            self._log_value(name=name, value=data, step=step)

    def log(self, population: Population) -> None:
        fitnesses = [er.fitness for er in population.evaluation_results]
        self._log_values(name='generation/fitness', values=fitnesses, step=population.generation)

        genome_ids = [er.genome_id for er in population.evaluation_results]
        genomes = [population.genomes[genome_id] for genome_id in genome_ids]

        ages = [genome.age for genome in genomes]
        self._log_values(name='generation/age', values=ages, step=population.generation)

        for name, data in population.logging_data:
            self._log_unknown(name=name, data=data, step=population.generation)
        population.logging_data.clear()
