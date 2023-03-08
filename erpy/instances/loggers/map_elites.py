from dataclasses import dataclass
from typing import Type, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

from erpy.framework.ea import EAConfig
from erpy.instances.loggers.wandb_logger import WandBLoggerConfig, WandBLogger, wandb_log_value, \
    wandb_log_values
from erpy.instances.populations.map_elites import MAPElitesPopulation


@dataclass
class MAPElitesLoggerConfig(WandBLoggerConfig):
    archive_dimension_labels: List[str]
    normalize_heatmaps: bool

    @property
    def logger(self) -> Type[WandBLogger]:
        return MAPElitesLogger


class MAPElitesLogger(WandBLogger):
    def __init__(self, config: EAConfig):
        super(MAPElitesLogger, self).__init__(config=config)

    @property
    def config(self) -> MAPElitesLoggerConfig:
        return super().config

    def _log_evaluation_result_data(self, population: MAPElitesPopulation) -> None:
        population.evaluation_results = [cell.evaluation_result for cell in population.archive.values()]
        super()._log_evaluation_result_data(population)
        population.evaluation_results.clear()

    def log(self, population: MAPElitesPopulation) -> None:
        # Log failed episodes
        super()._log_failures(population=population)

        wandb_log_value(run=self.run,
                        name='generation/num_evaluations',
                        value=population.num_evaluations,
                        step=population.generation)

        # Log evaluation results loggings
        self._log_evaluation_result_data(population=population)

        # Log archive fitnesses
        fitnesses = [cell.evaluation_result.fitness for cell in population.archive.values()]
        wandb_log_values(run=self.run, name='generation/fitness', values=fitnesses, step=population.generation)

        # Log archive ages
        ages = [cell.genome.age for cell in population.archive.values()]
        wandb_log_values(run=self.run, name='generation/age', values=ages, step=population.generation)

        # Log archive coverage
        coverage = population.coverage
        wandb_log_value(run=self.run, name='generation/archive_coverage', value=coverage, step=population.generation)

        # Log 2D heatmaps
        example_er = list(population.archive.values())[0].evaluation_result
        attributes_to_log = ["fitness"]
        try:
            attributes_to_log += list(example_er.info["ArchiveLoggerCallback"].keys())
        except KeyError:
            pass

        if isinstance(population, MAPElitesPopulation) and len(population.config.archive_dimensions) == 2:
            x_dim, y_dim = population.config.archive_dimensions
            grids = np.ones((len(attributes_to_log), y_dim, x_dim)) * np.inf

            for cell_index, cell in population.archive.items():
                x, y = cell_index
                for i, attribute in enumerate(attributes_to_log):
                    try:
                        value = cell.evaluation_result.__getattribute__(attribute)
                    except AttributeError:
                        value = cell.evaluation_result.info["ArchiveLoggerCallback"][attribute]
                    if abs(value) != np.inf:
                        grids[i, y, x] = value

            for grid, attribute in zip(grids, attributes_to_log):
                useful = grid != np.inf
                mask = grid == np.inf

                if self.config.normalize_heatmaps and attribute == "fitness":
                    grid[useful] /= np.max(grid[useful])

                ax = sns.heatmap(grid, mask=mask, linewidth=0., xticklabels=[0] + [None] * (x_dim - 2) + [1],
                                 yticklabels=[0] + [None] * (y_dim - 2) + [1],
                                 vmin=np.min(grid[useful]),
                                 vmax=np.max(grid[useful]))
                ax.set_xlabel(self.config.archive_dimension_labels[0])
                ax.set_ylabel(self.config.archive_dimension_labels[1])
                ax.invert_yaxis()

                self.run.log({f'archive/{attribute}': wandb.Image(ax)}, step=population.generation)
                plt.close()