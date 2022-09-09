from dataclasses import dataclass

import numpy as np

from base.ea import EAConfig
from base.evaluator import EvaluationResult
from base.genome import RobotGenome


@dataclass
class MAPElitesCell:
    genome: RobotGenome
    evaluation_result: EvaluationResult
    should_save: bool = False


@dataclass
class MAPElitesConfig(EAConfig):
    random_state: np.random.RandomState
    initial_archive_coverage: float
    initialisation_time_limit: float
