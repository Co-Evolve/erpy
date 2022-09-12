from __future__ import annotations

import abc
from typing import cast

import numpy as np

import erpy.base.specification as spec


class Phenome(metaclass=abc.ABCMeta):
    def __init__(self, specification: spec.Specification) -> None:
        self._specification = specification

    @property
    def specification(self) -> spec.Specification:
        return self._specification


class Robot(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, specification: spec.RobotSpecification) -> None:
        super().__init__(specification=specification)
        self._morphology = None
        self._controller = None

    @abc.abstractmethod
    def _build(self) -> None:
        raise NotImplementedError

    @property
    def morphology(self) -> Morphology:
        if self._morphology is None:
            self._build()
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
        if self._controller is None:
            self._build()
        return self._controller

    @property
    def specification(self) -> spec.RobotSpecification:
        return super().specification

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return self.controller(observations)


class Morphology(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, specification: spec.MorphologySpecification):
        super().__init__(specification=specification)


class Controller(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, specification: spec.ControllerSpecification):
        super().__init__(specification=specification)

    @abc.abstractmethod
    def __call__(self, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError
