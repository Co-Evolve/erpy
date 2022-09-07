from dataclasses import dataclass
from typing import Callable

import numpy as np

from erpy.base.ea import EAConfig
from erpy.base.evaluator import EvaluationResult
from erpy.base.genomes import Genome


@dataclass
class MAPElitesCell:
    genome: Genome
    evaluation_result: EvaluationResult
    should_save: bool = False


@dataclass
class MAPElitesConfig(EAConfig):
    random_state: np.random.RandomState
    genome_generator: Callable[[int], Genome]
    offspring_generator: Callable[[Genome, Callable[[], int]], Genome]
    morphological_innovation_protection: bool
    initial_archive_coverage: float
    initialisation_time_limit: float
    re_evaluate_checkpoint: bool
