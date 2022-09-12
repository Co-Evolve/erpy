from dataclasses import dataclass

from erpy.base.evaluator import EvaluationResult
from erpy.base.genome import RobotGenome


@dataclass
class MAPElitesCell:
    genome: RobotGenome
    evaluation_result: EvaluationResult
    should_save: bool = False
