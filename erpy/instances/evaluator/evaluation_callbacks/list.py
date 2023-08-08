from typing import List, Dict, Optional, Union, Any

import numpy as np

from erpy.framework.ea import EAConfig
from erpy.framework.environment import EnvironmentConfig, Environment
from erpy.framework.evaluator import EvaluationCallback, EvaluatorConfig, EvaluationResult
from erpy.framework.genome import Genome
from erpy.framework.phenome import Robot


class EvaluationCallbackList(EvaluationCallback):
    def __init__(self, evaluation_callbacks: List[EvaluationCallback]):
        super().__init__()
        self.callbacks = evaluation_callbacks

    @property
    def config(self) -> EvaluatorConfig:
        return self._config

    def before_evaluation(self, config: EAConfig,  shared_callback_data: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.before_evaluation(config=config,  shared_callback_data=shared_callback_data)

    def after_evaluation(self) -> None:
        for callback in self.callbacks:
            callback.after_evaluation()

    def from_env(self, env: Environment) -> None:
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

    def before_step(self, observations: Dict[str, np.ndarray], actions: np.ndarray) -> None:
        for callback in self.callbacks:
            callback.before_step(observations=observations, actions=actions)

    def after_step(self, observations: Dict[str, np.ndarray], actions: np.ndarray, rewards: Union[float, np.ndarray],
                   info: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callbacks:
            callback.after_step(observations=observations, actions=actions, rewards=rewards, info=info)

    def update_evaluation_result(self, evaluation_result: EvaluationResult) -> None:
        for callback in self.callbacks:
            callback.update_evaluation_result(evaluation_result)

    def update_environment_config(self, environment_config: EnvironmentConfig) -> None:
        for callback in self.callbacks:
            callback.update_environment_config(environment_config=environment_config)
