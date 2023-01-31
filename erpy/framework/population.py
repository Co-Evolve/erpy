from __future__ import annotations

import abc
from dataclasses import dataclass
from itertools import count
from typing import List, Dict, Type, Any, TYPE_CHECKING

import erpy.base.evaluator as evaluator
import erpy.base.genome as genome

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class PopulationConfig:
    population_size: int

    @property
    @abc.abstractmethod
    def population(self) -> Type[Population]:
        raise NotImplementedError


class Population(metaclass=abc.ABCMeta):
    def __init__(self, config: EAConfig) -> None:
        self._ea_config = config
        self._config = config.population_config

        self.generation = 0
        self.num_evaluations = 0
        self._logging_data: Dict[str, Any] = dict()
        self._saving_data: Dict[str, Any] = dict()
        self._genome_indexer = count(0)
        self._genomes: Dict[int, genome.Genome] = dict()

        self._to_maintain: List[int] = []
        self._to_reproduce: List[int] = []
        self._to_evaluate: List[int] = []
        self._under_evaluation: List[int] = []

        # This should hold the evaluation result of every genome in genomes after Evaluation
        self.evaluation_results: List[evaluator.EvaluationResult] = []

    @property
    def logging_data(self) -> Dict[str, Any]:
        return self._logging_data

    @property
    def saving_data(self) -> Dict[str, Any]:
        return self._saving_data

    @saving_data.setter
    def saving_data(self, data: Dict[str, Any]) -> None:
        self._saving_data = data

    @property
    def config(self) -> PopulationConfig:
        return self._config

    @property
    def ea_config(self) -> EAConfig:
        return self._ea_config

    @property
    def population_size(self) -> int:
        return self.config.population_size

    @property
    def genomes(self) -> Dict[int, genome.Genome]:
        return self._genomes

    @property
    def to_maintain(self) -> List[int]:
        return self._to_maintain

    @property
    def to_reproduce(self) -> List[int]:
        return self._to_reproduce

    @to_reproduce.setter
    def to_reproduce(self, to_reproduce: List[int]) -> None:
        self._to_reproduce = to_reproduce

    @property
    def to_evaluate(self) -> List[int]:
        return self._to_evaluate

    @property
    def under_evaluation(self) -> List[int]:
        return self._under_evaluation

    def before_evaluation(self) -> None:
        pass

    def after_evaluation(self) -> None:
        self.num_evaluations += len(self.evaluation_results)

    def get_next_child_id(self) -> int:
        return next(self._genome_indexer)
