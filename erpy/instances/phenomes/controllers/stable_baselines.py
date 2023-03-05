from typing import Dict, Any, Type, Union, Optional, Tuple

import numpy as np
import wandb
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from erpy.framework import evaluator
from erpy.framework.environment import Environment
from erpy.framework.phenome import Controller
from erpy.framework.specification import ControllerSpecification, RobotSpecification


class SBControllerSpecification(ControllerSpecification):
    def __init__(self,
                 algorithm: Type[BaseAlgorithm],
                 stable_baseline_model_arguments: Dict[str, Any]):
        super().__init__()
        self.algorithm = algorithm
        self.sb_model_arguments = stable_baseline_model_arguments

        self.path_to_model = None
        self.path_to_tensorboard_logs = None


class SBController(Controller):
    def __init__(self, specification: RobotSpecification):
        super().__init__(specification=specification)
        self.model: BaseAlgorithm = None

    @property
    def controller_specification(self) -> SBControllerSpecification:
        return super().controller_specification

    def _initialise_model(self, environment: Environment) -> None:
        if self.model is None:
            if self.controller_specification.path_to_tensorboard_logs is None:
                self.controller_specification.path_to_tensorboard_logs = f"/tmp/erpy/runs/{wandb.run.id}/"
            self.model = self.controller_specification.algorithm(
                env=environment,
                verbose=1,
                tensorboard_log=self.controller_specification.path_to_tensorboard_logs,
                **self.controller_specification.sb_model_arguments,
            )

            if self.controller_specification.path_to_model is not None:
                self.model = self.controller_specification.algorithm.load(
                    path=self.controller_specification.path_to_model,
                    tensorboard_log=self.controller_specification.path_to_tensorboard_logs)

    def set_environment(self, environment: Environment) -> None:
        self._initialise_model(environment=environment)
        if self.model.env is None or self.model.n_envs != environment.num_envs:
            self.model = self.controller_specification.algorithm.load(path=self.controller_specification.path_to_model,
                                                                      tensorboard_log=self.controller_specification.path_to_tensorboard_logs,
                                                                      env=environment)
        else:
            self.model.set_env(env=environment)

    def __call__(self, observations: Union[np.ndarray, Dict[str, np.ndarray]],
                 deterministic: bool = True, *args, **kwargs) -> np.ndarray:
        actions, _ = self.model.predict(observation=observations, deterministic=deterministic, *args, **kwargs)
        return actions

    def predict(self, observations: Union[np.ndarray, Dict[str, np.ndarray]], *args, **kwargs) -> Tuple[
        np.ndarray, Optional[np.ndarray]]:
        return self.model.predict(observation=observations, *args, **kwargs)

    def save(self, path: str) -> None:
        self.model.save(path=path)
        self.controller_specification.path_to_model = path

    def learn(self, total_timesteps: int, callback: Union[evaluator.EvaluationCallback, BaseCallback]) -> None:
        self.model.learn(total_timesteps=total_timesteps,
                         callback=callback,
                         reset_num_timesteps=False,  # This needs to be false to allow continual learning!
                         tb_log_name=wandb.run.id)
