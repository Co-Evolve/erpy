from __future__ import annotations

import abc

from erpy.base.genomes import Genome, RobotMorphologyGenome, RobotControllerGenome, RobotGenome


class Phenome(metaclass=abc.ABCMeta):
    def __init__(self, genome: Genome) -> None:
        self._genome = genome

    @property
    def genome(self) -> Genome:
        return self._genome

    @staticmethod
    @abc.abstractmethod
    def from_genome(genome: Genome) -> Phenome:
        raise NotImplementedError


class Robot(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, genome: RobotGenome, morphology: RobotMorphology, controller: RobotController):
        super().__init__(genome=genome)
        self.morphology = morphology
        self.controller = controller


class RobotMorphology(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, genome: RobotMorphologyGenome) -> None:
        super().__init__(genome=genome)


class RobotController(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, genome: RobotControllerGenome) -> None:
        super().__init__(genome=genome)

    @abc.abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError
