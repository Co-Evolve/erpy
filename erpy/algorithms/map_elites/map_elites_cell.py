from dataclasses import dataclass

from erpy.base.evaluator import EvaluationResult
from erpy.base.genome import Genome


@dataclass
class MAPElitesCell:
    genome: Genome
    evaluation_result: EvaluationResult
    should_save: bool = False
