from dataclasses import dataclass

from erpy.base.evaluator import EvaluatorConfig, Evaluator
from erpy.base.logger import LoggerConfig, Logger
from erpy.base.population import PopulationConfig, Population
from erpy.base.reproducer import ReproducerConfig, Reproducer
from erpy.base.saver import SaverConfig, Saver
from erpy.base.selector import SelectorConfig, Selector


@dataclass
class EAConfig:
    num_generations: int
    population_config: PopulationConfig
    evaluator_config: EvaluatorConfig
    selector_config: SelectorConfig
    reproducer_config: ReproducerConfig
    logger_config: LoggerConfig
    saver_config: SaverConfig

    @property
    def population(self) -> Population:
        return self.population_config.population(self)

    @property
    def evaluator(self) -> Evaluator:
        return self.evaluator_config.evaluator(self)

    @property
    def selector(self) -> Selector:
        return self.selector_config.selector(self)

    @property
    def reproducer(self) -> Reproducer:
        return self.reproducer_config.reproducer(self)

    @property
    def logger(self) -> Logger:
        return self.logger_config.logger(self)

    @property
    def saver(self) -> Saver:
        return self.saver_config.saver(self)


class EA:
    def __init__(self, config: EAConfig):
        self._config = config

        self.population = config.population
        self.evaluator = config.evaluator
        self.selector = config.selector
        self.reproducer = config.reproducer
        self.logger = config.logger
        self.saver = config.saver

    @property
    def config(self) -> EAConfig:
        return self._config

    def run(self) -> None:
        self.reproducer.initialise_population(self.population)

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
