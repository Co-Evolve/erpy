from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

from erpy.framework.evaluator import EvaluatorConfig, Evaluator, EvaluationResult
from erpy.framework.genome import Genome, DummyGenome
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

    cli_args: Optional[Dict[str, Any]] = None
    checkpoint_path: str = None
    num_generations: Optional[int] = None
    num_evaluations: Optional[int] = None

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
        is_done = False
        if self.config.num_generations is not None and self.population.generation >= self.config.num_generations:
            is_done = True
        if self.config.num_evaluations is not None and self.population.num_evaluations >= self.config.num_evaluations:
            is_done = True
        return is_done

    def run(self) -> None:
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

    def load_genomes(self, path: Optional[str] = None) -> List[Genome]:
        if path is not None:
            self.config.saver_config.save_path = path

        return self.saver.load()

    def analyze(self, path: Optional[str] = None) -> Tuple[List[Genome], Dict[int, EvaluationResult]]:
        genomes = self.load_genomes(path)

        return genomes, self.analyze_genomes(genomes)

    def analyze_specifications(self, specifications: List[RobotSpecification]) -> Tuple[
        List[Genome], Dict[int, EvaluationResult]]:
        genomes = [DummyGenome(genome_id=i, specification=specification) for i, specification in
                   enumerate(specifications)]
        return genomes, self.analyze_genomes(genomes=genomes)

    def analyze_genomes(self, genomes: List[Genome]) -> Dict[int, EvaluationResult]:
        self.population = self.config.population

        for genome in genomes:
            self.population.genomes[genome.genome_id] = genome
            self.population.to_evaluate.add(genome.genome_id)

        self.evaluator.evaluate(self.population)

        return self.population.evaluation_results
