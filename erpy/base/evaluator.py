import abc
from dataclasses import dataclass
from typing import Callable

from erpy.base.phenome import RobotMorphology
from erpy.base.population import Population
from erpy.base.types import Environment


@dataclass
class EvaluationResult:
    genome_id: int
    fitness: float


@dataclass
class EvaluatorConfig:
    make_env_fn: Callable[[RobotMorphology], Environment]
    num_eval_trials: int
    render: bool


class Evaluator(metaclass=abc.ABCMeta):
    def __init__(self, config: EvaluatorConfig) -> None:
        self._config = config

    @abc.abstractmethod
    def evaluate(self, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> EvaluatorConfig:
        return self._config
