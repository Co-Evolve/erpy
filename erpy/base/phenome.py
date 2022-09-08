from __future__ import annotations

import abc

import numpy as np

from base.specification import RobotSpecification, Specification, MorphologySpecification, ControllerSpecification


class Phenome(metaclass=abc.ABCMeta):
    def __init__(self, specification: Specification) -> None:
        self._specification = specification

    @property
    def specification(self) -> Specification:
        return self._specification


class Robot(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, specification: RobotSpecification):
        super().__init__(specification=specification)


class Morphology(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, specification: MorphologySpecification):
        super().__init__(specification=specification)


class Controller(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, specification: ControllerSpecification):
        super().__init__(specification=specification)

    @abc.abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError
