from dataclasses import dataclass
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

from algorithms.map_elites.population import MAPElitesPopulation
from loggers.wandb_logger import WandBLogger, WandBLoggerConfig


@dataclass
class MAPElitesLoggerConfig(WandBLoggerConfig):
    @property
    def logger(self) -> Type[WandBLogger]:
        return MAPElitesLogger


class MAPElitesLogger(WandBLogger):
    def __init__(self, config: MAPElitesLoggerConfig):
        super(MAPElitesLogger, self).__init__(config=config)

    def log(self, population: MAPElitesPopulation) -> None:
        # Log archive fitnesses
        fitnesses = [cell.evaluation_result.fitness for cell in population.archive.values()]
        super()._log_values(name='archive/fitness', values=fitnesses, step=population.generation)

        # Log archive ages
        ages = [cell.genome.age for cell in population.archive.values()]
        super()._log_values(name='archive/age', values=ages, step=population.generation)

        # Log archive coverage
        coverage = population.coverage

        # Log the archive
        x_dim, y_dim = population.config.archive_dimensions
        fitness_map = np.zeros((x_dim, y_dim))

        for descriptor, cell in population.archive.items():
            x, y = descriptor
            fitness = cell.evaluation_result.fitness
            if abs(fitness) != np.inf:
                fitness_map[x, y] = abs(fitness)

        mask = fitness_map == 0

        fitness_map /= np.max(fitness_map)
        fitness_map = 1 - fitness_map
        fitness_map[mask] = 0

        ax = sns.heatmap(fitness_map, mask=mask, linewidth=0., xticklabels=[0] + [None] * (x_dim - 2) + [1],
                         yticklabels=[0] + [None] * (y_dim - 2) + [1],
                         vmin=0,
                         vmax=np.max(fitness_map))
        ax.set_xlabel('Number of tendons')
        ax.set_ylabel('Average tendon length')
        ax.invert_yaxis()

        wandb.log({'archive/coverage': coverage,
                   'archive/normalized_fitness_map': wandb.Image(ax)}, step=population.generation)
        plt.close()
