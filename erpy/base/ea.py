from dataclasses import dataclass
from typing import Optional

from erpy.base.evaluator import Evaluator
from erpy.base.logger import Logger
from erpy.base.population import Population
from erpy.base.reproducer import Reproducer
from erpy.base.saver import Saver
from erpy.base.selector import Selector


@dataclass
class EAConfig:
    num_generations: int


class EA:
    def __init__(self, config: EAConfig, population: Population, evaluator: Evaluator, selector: Selector,
                 reproducer: Reproducer, logger: Optional[Logger] = None, saver: Optional[Saver] = None):
        self._config = config

        self.population = population
        self.evaluator = evaluator
        self.selector = selector
        self.reproducer = reproducer
        self.logger = logger
        self.saver = saver

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
