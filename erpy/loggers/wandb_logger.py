from __future__ import annotations

from dataclasses import dataclass
from typing import List, Type

import numpy as np
import wandb

from base.logger import Logger, LoggerConfig
from base.population import Population


@dataclass
class WandBLoggerConfig(LoggerConfig):
    project_name: str
    group: str
    tags: List[str]

    @property
    def logger(self) -> Type[WandBLogger]:
        return WandBLogger


class WandBLogger(Logger):
    def __init__(self, config: WandBLoggerConfig):
        super(WandBLogger, self).__init__(config=config)
        self.wandb = wandb.init(project=config.project_name,
                                group=config.group,
                                tags=config.tags)

    def _log_values(self, name: str, values: List[float], step: int) -> None:
        self.wandb.log({f'{name}_max': np.max(values),
                        f'{name}_mean': np.mean(values),
                        f'{name}_std': np.std(values)}, step=step)

    def log(self, population: Population) -> None:
        fitnesses = [er.fitness for er in population.evaluation_results]
        self._log_values(name='generation/fitness', values=fitnesses, step=population.generation)

        genome_ids = [er.genome_id for er in population.evaluation_results]
        genomes = [population.genomes[genome_id] for genome_id in genome_ids]

        ages = [genome.age for genome in genomes]
        self._log_values(name='generation/age', values=ages, step=population.generation)
