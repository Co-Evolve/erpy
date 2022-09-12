from __future__ import annotations

import abc
from dataclasses import dataclass
from itertools import count
from typing import List, Dict, Type

import erpy.base.evaluator as evaluator
import erpy.base.genome as genome


@dataclass
class PopulationConfig:
    population_size: int

    @property
    @abc.abstractmethod
    def population(self) -> Type[Population]:
        raise NotImplementedError


class Population(metaclass=abc.ABCMeta):
    def __init__(self, config: PopulationConfig) -> None:
        self._config = config

        self.generation = 0
        self._genome_indexer = count(0)
        self._genomes: Dict[int, genome.Genome] = {}

        self._to_maintain: List[int] = []
        self._to_reproduce: List[int] = []
        self._to_evaluate: List[int] = []
        self._under_evaluation: List[int] = []

        # This should hold the evaluation result of every genome in genomes after Evaluation
        self.evaluation_results: List[evaluator.EvaluationResult] = []

    @property
    def config(self) -> PopulationConfig:
        return self._config

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
        pass

    def get_next_child_id(self) -> int:
        return next(self._genome_indexer)
