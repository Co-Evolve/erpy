from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from base.evaluator import EvaluatorConfig
from base.genome import GenomeConfig
from base.logger import LoggerConfig
from base.population import PopulationConfig
from base.reproducer import ReproducerConfig
from base.saver import SaverConfig
from base.selector import SelectorConfig


@dataclass
class EAConfig:
    num_generations: int
    genome_config: GenomeConfig
    population_config: PopulationConfig
    evaluator_config: EvaluatorConfig
    selector_config: SelectorConfig
    reproducer_config: ReproducerConfig
    logger_config: LoggerConfig
    saver_config: SaverConfig


class EA:
    def __init__(self, config: EAConfig):
        self._config = config

        self.population = config.population_config.population(config.population_config)
        self.evaluator = config.evaluator_config.evaluator(config.evaluator_config)
        self.selector = config.selector_config.selector(config.selector_config)
        self.reproducer = config.reproducer_config.reproducer(config.reproducer_config)
        self.logger = config.logger_config.logger(config.logger_config)
        self.saver = config.saver_config.saver(config.saver_config)

    @property
    def config(self) -> EAConfig:
        return self._config

    def run(self) -> None:
        for generation in range(self._config.num_generations):
            self.population.generation = generation
            self.reproducer.reproduce(population=self.population)
            self.population.before_evaluation()
            self.evaluator.evaluate(population=self.population)
            self.population.after_evaluation()

            if self.logger is not None:
                self.logger.log(population=self.population)
            if self.saver is not None:
                self.saver.save(population=self.population)

            self.selector.select(population=self.population)
