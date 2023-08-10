from dataclasses import dataclass
from typing import Any, Dict, List

from erpy.framework.evaluator import EvaluationResult, Evaluator, EvaluatorConfig
from erpy.framework.genome import Genome
from erpy.framework.logger import Logger, LoggerConfig
from erpy.framework.population import Population, PopulationConfig
from erpy.framework.reproducer import Reproducer, ReproducerConfig
from erpy.framework.saver import Saver, SaverConfig
from erpy.framework.selector import Selector, SelectorConfig
from erpy.framework.specification import RobotSpecification
from erpy.instances.genome.dummy import DummyGenome


@dataclass
class EAConfig:
    population_config: PopulationConfig
    evaluator_config: EvaluatorConfig
    selector_config: SelectorConfig
    reproducer_config: ReproducerConfig
    logger_config: LoggerConfig
    saver_config: SaverConfig

    extra_args: Dict[str, Any]
    num_generations: int | None
    num_evaluations: int | None

    @property
    def population(
            self
            ) -> Population:
        return self.population_config.population(self)

    @property
    def evaluator(
            self
            ) -> Evaluator:
        return self.evaluator_config.evaluator(self)

    @property
    def selector(
            self
            ) -> Selector:
        return self.selector_config.selector(self)

    @property
    def reproducer(
            self
            ) -> Reproducer:
        return self.reproducer_config.reproducer(self)

    @property
    def logger(
            self
            ) -> Logger:
        return self.logger_config.logger(self)

    @property
    def saver(
            self
            ) -> Saver:
        return self.saver_config.saver(self)


class EA:
    def __init__(
            self,
            config: EAConfig
            ):
        self._config = config

        self.population = self.config.population
        self.selector = self.config.selector
        self.reproducer = self.config.reproducer
        self.logger = self.config.logger
        self.saver = self.config.saver
        self.evaluator = self.config.evaluator

    @property
    def config(
            self
            ) -> EAConfig:
        return self._config

    def is_done(
            self
            ) -> bool:
        is_done = False
        if self.config.num_generations is not None and self.population.generation >= self.config.num_generations:
            is_done = True
        if self.config.num_evaluations is not None and self.population.num_evaluations >= self.config.num_evaluations:
            is_done = True
        return is_done

    def optimize(
            self
            ) -> None:
        self.reproducer.initialise_population(self.population)

        while not self.is_done():
            self.population.before_reproduction()
            self.reproducer.reproduce(population=self.population)
            self.population.after_reproduction()

            self.population.before_evaluation()
            self.evaluator.evaluate(population=self.population)
            self.population.after_evaluation()

            self.population.before_logging()
            self.logger.log(population=self.population)
            self.population.after_logging()

            self.population.before_saving()
            self.saver.save(population=self.population)
            self.population.after_saving()

            self.population.before_selection()
            self.selector.select(population=self.population)
            self.population.after_selection()

            self.population.generation += 1

    def analyze_specifications(
            self,
            specifications: List[RobotSpecification]
            ) -> List[EvaluationResult]:
        genomes = [DummyGenome(genome_id=i, specification=specification) for i, specification in
                   enumerate(specifications)]
        return self.analyze_genomes(genomes=genomes)

    def analyze_genomes(
            self,
            genomes: List[Genome]
            ) -> List[EvaluationResult]:
        self.population = self.config.population

        for genome in genomes:
            self.population.genomes[genome.genome_id] = genome
            self.population.to_evaluate.add(genome.genome_id)

        self.evaluator.evaluate(self.population)

        return self.population.evaluation_results
