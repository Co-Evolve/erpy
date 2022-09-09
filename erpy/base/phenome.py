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
        self._morphology = specification.morphology_specification.to_phenome()
        self._controller = specification.controller_specification.to_phenome()

    @property
    def morphology(self) -> Morphology:
        return self._morphology

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Called at the start of every episode.
        Use this to reset the controller if needed.
        :return:
        """
        raise NotImplementedError

    @property
    def controller(self) -> Controller:
        return self._controller

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return self.controller(observations)


class Morphology(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, specification: MorphologySpecification):
        super().__init__(specification=specification)


class Controller(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, specification: ControllerSpecification):
        super().__init__(specification=specification)

    @abc.abstractmethod
    def __call__(self, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError
