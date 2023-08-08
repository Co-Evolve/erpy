from dataclasses import dataclass
from typing import List, Dict, Any

from erpy.framework.evaluator import EvaluatorConfig, Evaluator, EvaluationResult
from erpy.framework.genome import Genome
from erpy.framework.logger import LoggerConfig, Logger
from erpy.framework.population import PopulationConfig, Population
from erpy.framework.reproducer import ReproducerConfig, Reproducer
from erpy.framework.saver import SaverConfig, Saver
from erpy.framework.selector import SelectorConfig, Selector
from erpy.framework.specification import RobotSpecification


@dataclass
class EAConfig:
    population_config: PopulationConfig
    evaluator_config: EvaluatorConfig
    selector_config: SelectorConfig
    reproducer_config: ReproducerConfig
    logger_config: LoggerConfig
    saver_config: SaverConfig

    extra_args: Dict[str, Any]

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

        self.population = self.config.population
        self.selector = self.config.selector
        self.reproducer = self.config.reproducer
        self.logger = self.config.logger
        self.saver = self.config.saver
        self.evaluator = self.config.evaluator

    @property
    def config(self) -> EAConfig:
        return self._config

    def is_done(self) -> bool:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def analyze_specifications(self, specifications: List[RobotSpecification]) -> List[EvaluationResult]:
        raise NotImplementedError

    def analyze_genomes(self, genomes: List[Genome]) -> List[EvaluationResult]:
        raise NotImplementedError
