from __future__ import annotations

import pickle
from dataclasses import dataclass

from erpy.instances.algorithms.map_elites.types import PhenomeDescription
from erpy.framework.evaluator import EvaluationResult
from erpy.framework.genome import Genome


@dataclass
class MAPElitesCell:
    descriptor: PhenomeDescription
    genome: Genome
    evaluation_result: EvaluationResult
    should_save: bool = False

    def save(self, path: str) -> None:
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> MAPElitesCell:
        with open(path, 'rb') as handle:
            return pickle.load(handle)
