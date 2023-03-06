from __future__ import annotations

import abc
from typing import Union, Dict, Tuple, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import erpy.framework.environment as env
import erpy.framework.evaluator as evaluator
import erpy.framework.specification as spec


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

    @property
    def morphology(self) -> Morphology:
        if self._morphology is None:
            self._morphology = self._build_morphology()
        return self._morphology

    @property
    def controller(self) -> Controller:
        if self._controller is None:
            self._controller = self._build_controller()
        return self._controller

    @abc.abstractmethod
    def _build_morphology(self) -> Morphology:
        raise NotImplementedError

    @abc.abstractmethod
    def _build_controller(self) -> Controller:
        raise NotImplementedError

    def reset(self) -> None:
        """
        Called at the start of every episode.
        Use this to reset the controller if needed.
        :return:
        """
        pass

    @property
    def specification(self) -> spec.RobotSpecification:
        return super().specification

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return self.controller(observations)


class Morphology(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, specification: spec.RobotSpecification):
        super().__init__(specification=specification)

    @property
    def specification(self) -> spec.RobotSpecification:
        return self._specification

    @property
    def controller_specification(self) -> spec.ControllerSpecification:
        return self.specification.controller_specification

    @property
    def morphology_specification(self) -> spec.MorphologySpecification:
        return self.specification.morphology_specification


class Controller(Phenome, metaclass=abc.ABCMeta):
    def __init__(self, specification: spec.RobotSpecification):
        super().__init__(specification=specification)
        self._environment = None

    @property
    def specification(self) -> spec.RobotSpecification:
        return self._specification

    @property
    def controller_specification(self) -> spec.ControllerSpecification:
        return self.specification.controller_specification

    @property
    def morphology_specification(self) -> spec.MorphologySpecification:
        return self.specification.morphology_specification

    def set_environment(self, environment: env.Environment) -> None:
        self._environment = environment

    @abc.abstractmethod
    def __call__(self, observations: Union[np.ndarray, Dict[str, np.ndarray]],
                 deterministic: bool = True) -> np.ndarray:
        raise NotImplementedError

    def predict(self, observations: Union[np.ndarray, Dict[str, np.ndarray]], *args, **kwargs) -> Tuple[
        np.ndarray, Optional[np.ndarray]]:
        return self(observations=observations, *args, **kwargs), None

    def learn(self, total_timesteps: int, callback: Union[evaluator.EvaluationCallback, BaseCallback]) -> None:
        pass
