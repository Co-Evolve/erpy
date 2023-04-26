from __future__ import annotations

import abc
from dataclasses import dataclass
from itertools import count
from typing import Dict, Type, Any, TYPE_CHECKING, Set, List

import erpy.framework.evaluator as evaluator
import erpy.framework.genome as genome

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

        self._to_maintain: Set[int] = set()
        self._to_reproduce: Set[int] = set()
        self._to_evaluate: Set[int] = set()
        self._under_evaluation: Set[int] = set()

        # This should hold the evaluation result of every genome in genomes after Evaluation
        self.evaluation_results: List[evaluator.EvaluationResult] = list()
        self._all_time_best_evaluation_result: evaluator.EvaluationResult = None

    @property
    def logging_data(self) -> Dict[str, Any]:
        return self._logging_data

    @property
    def all_time_best_evaluation_result(self) -> evaluator.EvaluationResult:
        return self._all_time_best_evaluation_result

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
    def genomes(self) -> Dict[int, genome.Genome]:
        return self._genomes

    @property
    def to_maintain(self) -> Set[int]:
        return self._to_maintain

    @property
    def to_reproduce(self) -> Set[int]:
        return self._to_reproduce

    @property
    def to_evaluate(self) -> Set[int]:
        return self._to_evaluate

    @property
    def under_evaluation(self) -> Set[int]:
        return self._under_evaluation

    def before_reproduction(self) -> None:
        pass

    def after_reproduction(self) -> None:
        self.to_reproduce.clear()

        genome_ids_to_keep = self.to_maintain.union(self.to_reproduce).union(self.to_evaluate).union(
            self.under_evaluation)
        current_genome_ids = set(self.genomes.keys())
        to_remove = current_genome_ids - genome_ids_to_keep
        for key in to_remove:
            self.genomes.pop(key)

    def before_logging(self) -> None:
        pass

    def after_logging(self) -> None:
        self.logging_data.clear()

    def before_saving(self) -> None:
        pass

    def after_saving(self) -> None:
        self.saving_data.clear()

    def before_selection(self) -> None:
        pass

    def after_selection(self) -> None:
        pass

    def before_evaluation(self) -> None:
        self.evaluation_results.clear()

    def after_evaluation(self) -> None:
        self.num_evaluations += len(self.evaluation_results)

        self.to_evaluate.clear()

        for evaluation_result in self.evaluation_results:
            self.genomes[evaluation_result.genome.genome_id] = evaluation_result.genome
            self.under_evaluation.discard(evaluation_result.genome.genome_id)
            if self.all_time_best_evaluation_result is None or \
                    self.all_time_best_evaluation_result.fitness < evaluation_result.fitness:
                self._all_time_best_evaluation_result = evaluation_result

    def get_next_child_id(self) -> int:
        return next(self._genome_indexer)
