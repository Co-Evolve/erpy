from __future__ import annotations

import abc
from dataclasses import dataclass
from itertools import count
from typing import List, Dict, Type

from base.evaluator import EvaluationResult
from base.genome import Genome


@dataclass
class PopulationConfig:
    @property
    @abc.abstractmethod
    def population(self) -> Type[Population]:
        raise NotImplementedError


class Population(metaclass=abc.ABCMeta):
    def __init__(self, config: PopulationConfig) -> None:
        self._config = config

        self.generation = 0
        self._genome_indexer = count(0)
        self._genomes: Dict[int, Genome] = {}

        self._to_maintain: List[int] = []
        self._to_reproduce: List[int] = []
        self._to_evaluate: List[int] = []
        self._under_evaluation: List[int] = []

        # This should hold the evaluation result of every genome in genomes after Evaluation
        self.evaluation_results: List[EvaluationResult] = []

    @property
    def config(self) -> PopulationConfig:
        return self._config

    @property
    def genomes(self) -> Dict[int, Genome]:
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
