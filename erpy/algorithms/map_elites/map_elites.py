from dataclasses import dataclass
from typing import Callable

import numpy as np

from base.ea import EAConfig
from base.evaluator import EvaluationResult
from base.genome import Genome


@dataclass
class MAPElitesCell:
    genome: Genome
    evaluation_result: EvaluationResult
    should_save: bool = False


@dataclass
class MAPElitesConfig(EAConfig):
    random_state: np.random.RandomState
    morphological_innovation_protection: bool
    initial_archive_coverage: float
    initialisation_time_limit: float
    re_evaluate_checkpoint: bool
