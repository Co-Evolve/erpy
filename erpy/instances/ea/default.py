from dataclasses import dataclass
from typing import List

from erpy.framework.ea import EAConfig, EA
from erpy.framework.evaluator import EvaluationResult
from erpy.framework.genome import Genome
from erpy.framework.specification import RobotSpecification
from erpy.instances.evaluator.default import DefaultEvaluatorConfig, DefaultEvaluator
from erpy.instances.genome.dummy import DummyGenome
from erpy.instances.logger.default import DefaultLoggerConfig, DefaultLogger
from erpy.instances.population.default import DefaultPopulationConfig, DefaultPopulation
from erpy.instances.reproducer.default import DefaultReproducerConfig, DefaultReproducer
from erpy.instances.saver.default import DefaultSaverConfig, DefaultSaver
from erpy.instances.selector.default import DefaultSelectorConfig, DefaultSelector


@dataclass
class DefaultEAConfig(EAConfig):
    population_config: DefaultPopulationConfig
    evaluator_config: DefaultEvaluatorConfig
    selector_config: DefaultSelectorConfig
    reproducer_config: DefaultReproducerConfig
    logger_config: DefaultLoggerConfig
    saver_config: DefaultSaverConfig

    num_generations: int | None
    num_evaluations: int | None

    @property
    def population(self) -> DefaultPopulation:
        return super().population

    @property
    def evaluator(self) -> DefaultEvaluator:
        return super().evaluator

    @property
    def selector(self) -> DefaultSelector:
        return super().selector

    @property
    def reproducer(self) -> DefaultReproducer:
        return super().reproducer

    @property
    def logger(self) -> DefaultLogger:
        return super().logger

    @property
    def saver(self) -> DefaultSaver:
        return super().saver


class DefaultEA(EA):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(config)

    @property
    def config(self) -> DefaultEAConfig:
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

    def load_genomes(self, path: str | None = None) -> List[Genome]:
        if path is not None:
            self.config.saver_config.save_path = path

        return self.saver.load()

    def analyze(self, path: str | None = None) -> List[EvaluationResult]:
        genomes = self.load_genomes(path)

        return self.analyze_genomes(genomes)

    def analyze_specifications(self, specifications: List[RobotSpecification]) -> List[EvaluationResult]:
        genomes = [DummyGenome(genome_id=i, specification=specification) for i, specification in
                   enumerate(specifications)]
        return self.analyze_genomes(genomes=genomes)

    def analyze_genomes(self, genomes: List[Genome]) -> List[EvaluationResult]:
        self.population = self.config.population

        for genome in genomes:
            self.population.genomes[genome.genome_id] = genome
            self.population.to_evaluate.add(genome.genome_id)

        self.evaluator.evaluate(self.population)

        return self.population.evaluation_results
