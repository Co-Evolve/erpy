from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Iterable, Dict, Any, Type, TYPE_CHECKING, Optional, Union

import gymnasium as gym
import numpy as np

import erpy.framework.environment as environment
import erpy.framework.genome as genome
import erpy.framework.phenome as phenome
import erpy.framework.population as population

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class EvaluationResult:
    genome: genome.Genome
    fitness: float
    info: Dict[str, Any]


@dataclass
class EvaluatorConfig(metaclass=abc.ABCMeta):
    environment_config: environment.EnvironmentConfig
    robot: Type[phenome.Robot]
    reward_aggregator: Callable[[Iterable[float]], float]
    episode_aggregator: Callable[[Iterable[float]], float]
    num_eval_episodes: int
    hard_episode_reset: bool
    callback: Optional[EvaluationCallback]

    @property
    @abc.abstractmethod
    def evaluator(self) -> Type[Evaluator]:
        raise NotImplementedError


class Evaluator(metaclass=abc.ABCMeta):
    def __init__(self, config: EAConfig) -> None:
        self._ea_config = config
        self._config = config.evaluator_config

    @abc.abstractmethod
    def evaluate(self, population: population.Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> EvaluatorConfig:
        return self._config


class EvaluationActor(metaclass=abc.ABCMeta):
    def __init__(self, config: EAConfig) -> None:
        self._ea_config = config
        self._config = config.evaluator_config
        self._callback = self._initialise_callback()

    def _initialise_callback(self) -> EvaluationCallback:
        if self.config.callback is not None:
            return self.config.callback
        else:
            return EvaluationCallback()

    @property
    def config(self) -> EvaluatorConfig:
        return self._config

    @abc.abstractmethod
    def evaluate(self, genome: genome.Genome) -> EvaluationResult:
        raise NotImplementedError


class EvaluationCallback(metaclass=abc.ABCMeta):
    def __init__(self, name: Optional[str] = None) -> None:
        self._ea_config = None
        self._config = None

        self._shared_callback_data = None

        if name is None:
            name = self.__class__.__name__
        self._name = name

    @property
    def shared_callback_data(self) -> Dict[any, str]:
        return self._shared_callback_data

    @property
    def ea_config(self) -> EAConfig:
        return self._ea_config

    @property
    def config(self) -> EvaluatorConfig:
        return self._config

    def __call__(self, *args, **kwargs):
        pass

    def before_evaluation(self, config: EAConfig, shared_callback_data: Dict[str, Any]) -> None:
        self._ea_config = config
        self._config = config.evaluator_config
        self._shared_callback_data = shared_callback_data

    def after_evaluation(self) -> None:
        pass

    def from_env(self, env: gym.Env) -> None:
        pass

    def before_episode(self) -> None:
        pass

    def after_episode(self) -> None:
        pass

    def from_robot(self, robot: phenome.Robot) -> None:
        pass

    def from_genome(self, genome: genome.Genome) -> None:
        pass

    def before_step(self, observations: Dict[str, np.ndarray], actions: np.ndarray) -> None:
        pass

    def after_step(self, observations: Dict[str, np.ndarray], actions: np.ndarray,
                   rewards: Union[float, np.ndarray], info: Optional[Dict[str, Any]] = None) -> None:
        pass

    def update_evaluation_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        return evaluation_result

    def update_environment_config(self, environment_config: environment.EnvironmentConfig) -> None:
        pass

    @property
    def name(self) -> str:
        return self._name
