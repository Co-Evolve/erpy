from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable

from dm_control import composer
from stable_baselines3.common.monitor import Monitor

from erpy.base.environment import EnvironmentConfig, Environment
from erpy.interfaces.mujoco.gym_wrapper import DMC2GymWrapper
from erpy.interfaces.mujoco.phenome import MJCRobot


def default_make_mjc_env(config: MJCEnvironmentConfig, robot: MJCRobot) -> Environment:
    task = config.task(config, robot.morphology)
    dm_env = composer.Environment(task=task,
                                  random_state=config.random_state,
                                  time_limit=config.simulation_time)
    env = DMC2GymWrapper(env=dm_env,
                         seed=config.seed,
                         from_pixels=False,
                         camera_ids=[0, 1]
                         )
    env = Monitor(env)
    return env


@dataclass
class MJCEnvironmentConfig(EnvironmentConfig, abc.ABC):
    @property
    @abc.abstractmethod
    def task(self) -> Callable[[MJCEnvironmentConfig, MJCRobot], composer.Task]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def simulation_time(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_substeps(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def time_scale(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def physics_time_delta(self) -> float:
        raise NotImplementedError

    @property
    def num_timesteps(self) -> int:
        return int(self.simulation_time / (self.physics_time_delta * self.time_scale) / self._num_substeps)

    def environment(self, robot: MJCRobot) -> Environment:
        return default_make_mjc_env(config=self, robot=robot)
