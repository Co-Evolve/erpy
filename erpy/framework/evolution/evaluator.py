from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

import erpy.framework.environment as environment
import erpy.framework.genome as genome
import erpy.framework.phenome as phenome
import erpy.framework.population as population
from erpy.framework.component import EAComponent, EAComponentConfig

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class EvaluationResult:
    genome: genome.Genome
    fitness: float
    info: Dict[str, Any]


@dataclass
class EvaluatorConfig(EAComponentConfig):
    environment_config: environment.EnvironmentConfig
    robot: Type[phenome.Robot]
    callback: EvaluationCallback | None

    @property
    @abc.abstractmethod
    def evaluator(
            self
            ) -> Type[Evaluator]:
        raise NotImplementedError


class Evaluator(EAComponent):
    def __init__(
            self,
            config: EAConfig
            ) -> None:
        super().__init__(config)

    @property
    def config(
            self
            ) -> EvaluatorConfig:
        return self.ea_config.evaluator_config

    @abc.abstractmethod
    def evaluate(
            self,
            population: population.Population
            ) -> None:
        raise NotImplementedError


class EvaluationActor(EAComponent):
    def __init__(
            self,
            config: EAConfig
            ) -> None:
        super().__init__(config)

        self._callback = self._initialise_callback()

    @property
    def config(
            self
            ) -> EvaluatorConfig:
        return self.ea_config.evaluator_config

    def _initialise_callback(
            self
            ) -> EvaluationCallback:
        if self.config.callback is not None:
            return self.config.callback
        else:
            return EvaluationCallback()

    @abc.abstractmethod
    def evaluate(
            self,
            genome: genome.Genome
            ) -> EvaluationResult:
        raise NotImplementedError


class EvaluationCallback(metaclass=abc.ABCMeta):
    def __init__(
            self
            ) -> None:
        self._ea_config: EAConfig | None = None
        self._config: EvaluatorConfig | None = None
        self._shared_callback_data: Dict[str, Any] | None = None

    @property
    def shared_callback_data(
            self
            ) -> Dict[str, Any]:
        return self._shared_callback_data

    @property
    def ea_config(
            self
            ) -> EAConfig:
        return self._ea_config

    @property
    def config(
            self
            ) -> EvaluatorConfig:
        return self._config

    def before_evaluation(
            self,
            config: EAConfig,
            shared_callback_data: Dict[str, Any]
            ) -> None:
        self._ea_config = config
        self._config = config.evaluator_config
        self._shared_callback_data = shared_callback_data

    def after_evaluation(
            self
            ) -> None:
        pass

    def from_env(
            self,
            env: gym.Env
            ) -> None:
        pass

    def before_episode(
            self
            ) -> None:
        pass

    def after_episode(
            self
            ) -> None:
        pass

    def from_robot(
            self,
            robot: phenome.Robot
            ) -> None:
        pass

    def from_genome(
            self,
            genome: genome.Genome
            ) -> None:
        pass

    def before_step(
            self,
            observations: ObsType,
            actions: ActType
            ) -> None:
        pass

    def after_step(
            self,
            observations: ObsType,
            actions: ActType,
            reward: Union[float, np.ndarray],
            info: Optional[Dict[str, Any]] = None
            ) -> None:
        pass

    def update_evaluation_result(
            self,
            evaluation_result: EvaluationResult
            ) -> None:
        pass

    def update_environment_config(
            self,
            environment_config: environment.EnvironmentConfig
            ) -> None:
        pass
