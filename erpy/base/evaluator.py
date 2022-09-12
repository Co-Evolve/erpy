from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Tuple, Iterable, List, Dict, Any, Type

import numpy as np

from erpy.base.genome import RobotGenome
from erpy.base.phenome import Robot
from erpy.base.population import Population
from erpy.base.types import Environment
from tasks.task_config import TaskConfig


@dataclass
class EvaluationResult:
    genome_id: int
    fitness: float
    info: Dict[str, Any]


@dataclass
class EvaluatorConfig(metaclass=abc.ABCMeta):
    make_env_fn: Callable[[TaskConfig, Robot], Tuple[Environment]]
    robot: Type[Robot]
    reward_aggregator: Callable[[Iterable[float]], float]
    episode_aggregator: Callable[[Iterable[float]], float]
    callbacks: List[Type[EvaluationCallback]]
    num_eval_episodes: int
    render: bool
    task_config: TaskConfig

    @property
    @abc.abstractmethod
    def evaluator(self) -> Type[Evaluator]:
        raise NotImplementedError


class Evaluator(metaclass=abc.ABCMeta):
    def __init__(self, config: EvaluatorConfig) -> None:
        self._config = config

    @abc.abstractmethod
    def evaluate(self, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> EvaluatorConfig:
        return self._config


class EvaluationActor(metaclass=abc.ABCMeta):
    def __init__(self, config: EvaluatorConfig) -> None:
        self._config = config
        self._callback_handler = EvaluationCallbackHandler(config=config)

    @property
    def config(self) -> EvaluatorConfig:
        return self._config

    @abc.abstractmethod
    def evaluate(self, genome: RobotGenome) -> EvaluationResult:
        raise NotImplementedError

    @property
    def callback_handler(self) -> EvaluationCallbackHandler:
        return self._callback_handler


class EvaluationCallback(metaclass=abc.ABCMeta):
    def __init__(self, name='str') -> None:
        self._name = name
        self._data = None

    def before_episode(self):
        pass

    def after_episode(self):
        pass

    def from_robot(self, robot: Robot):
        pass

    def from_genome(self, genome: RobotGenome):
        pass

    def before_step(self, observations, actions):
        pass

    def after_step(self, observations, actions):
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
    def __init__(self, config: EvaluatorConfig):
        self._config = config
        self.callbacks = []
        self.reset()

    def reset(self) -> None:
        self.callbacks = [callback() for callback in self._config.callbacks]

    def before_episode(self) -> None:
        for callback in self.callbacks:
            callback.before_episode()

    def after_episode(self) -> None:
        for callback in self.callbacks:
            callback.after_episode()

    def from_genome(self, genome: RobotGenome) -> None:
        for callback in self.callbacks:
            callback.from_genome(genome)

    def from_robot(self, robot: Robot) -> None:
        for callback in self.callbacks:
            callback.from_robot(robot=robot)

    def before_step(self, observations: np.ndarray, actions: np.ndarray) -> None:
        for callback in self.callbacks:
            callback.before_step(observations=observations, actions=actions)

    def after_step(self, observations: np.ndarray, actions: np.ndarray) -> None:
        for callback in self.callbacks:
            callback.after_step(observations=observations, actions=actions)

    def update_evaluation_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        for callback in self.callbacks:
            evaluation_result = callback.update_evaluation_result(evaluation_result)
        return evaluation_result
