from __future__ import annotations

import pickle
from dataclasses import dataclass

from erpy.framework.evaluator import EvaluationResult
from erpy.framework.genome import Genome
from erpy.instances.algorithms.map_elites.types import PhenomeDescription


@dataclass
class MAPElitesCell:
    descriptor: PhenomeDescription
    evaluation_result: EvaluationResult
    should_save: bool = False

    @property
    def genome(self) -> Genome:
        return self.evaluation_result.genome

    def save(self, path: str) -> None:
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> MAPElitesCell:
        with open(path, 'rb') as handle:
            return pickle.load(handle)
