from dataclasses import dataclass

from erpy.framework.ea import EA, EAConfig
from erpy.instances.evaluator.default import DefaultEvaluator, DefaultEvaluatorConfig
from erpy.instances.logger.default import DefaultLogger, DefaultLoggerConfig
from erpy.instances.population.default import DefaultPopulation, DefaultPopulationConfig
from erpy.instances.reproducer.default import DefaultReproducer, DefaultReproducerConfig
from erpy.instances.saver.default import DefaultSaver, DefaultSaverConfig
from erpy.instances.selector.default import DefaultSelector, DefaultSelectorConfig


@dataclass
class DefaultEAConfig(EAConfig):
    population_config: DefaultPopulationConfig
    evaluator_config: DefaultEvaluatorConfig
    selector_config: DefaultSelectorConfig
    reproducer_config: DefaultReproducerConfig
    logger_config: DefaultLoggerConfig
    saver_config: DefaultSaverConfig

    @property
    def population(
            self
            ) -> DefaultPopulation:
        return super().population

    @property
    def evaluator(
            self
            ) -> DefaultEvaluator:
        return super().evaluator

    @property
    def selector(
            self
            ) -> DefaultSelector:
        return super().selector

    @property
    def reproducer(
            self
            ) -> DefaultReproducer:
        return super().reproducer

    @property
    def logger(
            self
            ) -> DefaultLogger:
        return super().logger

    @property
    def saver(
            self
            ) -> DefaultSaver:
        return super().saver


class DefaultEA(EA):
    def __init__(
            self,
            config: EAConfig
            ) -> None:
        super().__init__(config)

    @property
    def config(
            self
            ) -> DefaultEAConfig:
        return super().config
