from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Iterable, List, Dict, Any, Type, TYPE_CHECKING, Optional

import gym
import numpy as np

from erpy.base.environment import EnvironmentConfig
from erpy.base.genome import Genome
from erpy.base.phenome import Robot
from erpy.base.population import Population

if TYPE_CHECKING:
    from erpy.base.ea import EAConfig


@dataclass
class EvaluationResult:
    genome_id: int
    fitness: float
    info: Dict[str, Any]


@dataclass
class EvaluatorConfig(metaclass=abc.ABCMeta):
    environment_config: EnvironmentConfig
    robot: Type[Robot]
    reward_aggregator: Callable[[Iterable[float]], float]
    episode_aggregator: Callable[[Iterable[float]], float]
    callbacks: List[Type[EvaluationCallback]]
    analyze_callbacks: List[Type[EvaluationCallback]]
    num_eval_episodes: int

    @property
    @abc.abstractmethod
    def evaluator(self) -> Type[Evaluator]:
        raise NotImplementedError


class Evaluator(metaclass=abc.ABCMeta):
    def __init__(self, config: EAConfig) -> None:
        self._ea_config = config
        self._config = config.evaluator_config

    @abc.abstractmethod
    def evaluate(self, population: Population, analyze: bool = False) -> None:
        raise NotImplementedError

    @property
    def config(self) -> EvaluatorConfig:
        return self._config


class EvaluationActor(metaclass=abc.ABCMeta):
    def __init__(self, config: EAConfig) -> None:
        self._ea_config = config
        self._config = config.evaluator_config
        self._callback_handler = EvaluationCallbackHandler(config=config)

    @property
    def config(self) -> EvaluatorConfig:
        return self._config

    @abc.abstractmethod
    def evaluate(self, genome: Genome, analyze: bool = False) -> EvaluationResult:
        raise NotImplementedError

    @property
    def callback_handler(self) -> EvaluationCallbackHandler:
        return self._callback_handler


class EvaluationCallback(metaclass=abc.ABCMeta):
    def __init__(self, config: EAConfig, name: str = 'unnamed') -> None:
        self._ea_config = config
        self._config = config.evaluator_config
        self._name = name
        self._data = None

    @property
    def config(self) -> EvaluatorConfig:
        return self._config

    def from_env(self, env: gym.Env) -> None:
        pass

    def before_episode(self):
        pass

    def after_episode(self):
        pass

    def from_robot(self, robot: Robot):
        pass

    def from_genome(self, genome: Genome):
        pass

    def before_step(self, observations, actions):
        pass

    def after_step(self, observations, actions, info=None):
        pass

    def update_evaluation_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        evaluation_result.info[self.name] = self.data
        return evaluation_result

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self):
        return self._data


class EvaluationCallbackHandler:
    def __init__(self, config: EAConfig):
        self._ea_config = config
        self._config = config.evaluator_config
        self.callbacks = []
        self.reset()

    @property
    def config(self) -> EvaluatorConfig:
        return self._config

    def reset(self, analyze: bool = False) -> None:
        if analyze:
            cbs = self.config.analyze_callbacks
        else:
            cbs = self.config.callbacks

        self.callbacks = [callback(config=self._ea_config) for callback in cbs]

    def from_env(self, env: gym.Env) -> None:
        for callback in self.callbacks:
            callback.from_env(env=env)

    def before_episode(self) -> None:
        for callback in self.callbacks:
            callback.before_episode()

    def after_episode(self) -> None:
        for callback in self.callbacks:
            callback.after_episode()

    def from_genome(self, genome: Genome) -> None:
        for callback in self.callbacks:
            callback.from_genome(genome)

    def from_robot(self, robot: Robot) -> None:
        for callback in self.callbacks:
            callback.from_robot(robot=robot)

    def before_step(self, observations: np.ndarray, actions: np.ndarray) -> None:
        for callback in self.callbacks:
            callback.before_step(observations=observations, actions=actions)

    def after_step(self, observations: np.ndarray, actions: np.ndarray, info: Optional[Dict] = None) -> None:
        for callback in self.callbacks:
            callback.after_step(observations=observations, actions=actions, info=info)

    def update_evaluation_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        for callback in self.callbacks:
            evaluation_result = callback.update_evaluation_result(evaluation_result)
        return evaluation_result
