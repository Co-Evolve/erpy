from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, List

from dm_control import composer
from stable_baselines3.common.monitor import Monitor

from erpy.base.environment import EnvironmentConfig, Environment
from erpy.interfaces.mujoco.gym_wrapper import DMC2GymWrapper
from erpy.interfaces.mujoco.phenome import MJCRobot, MJCMorphology


def default_make_mjc_env(config: MJCEnvironmentConfig, robot: MJCRobot, wrap2gym: bool = True) -> Environment:
    task = config.task(config, robot.morphology)
    env = composer.Environment(task=task,
                               random_state=config.random_state,
                               time_limit=config.simulation_time)

    if wrap2gym:
        env = DMC2GymWrapper(env=env,
                             seed=config.seed,
                             from_pixels=False,
                             camera_ids=config.camera_ids
                             )
        env = Monitor(env)
    return env


@dataclass
class MJCEnvironmentConfig(EnvironmentConfig, abc.ABC):
    @property
    @abc.abstractmethod
    def task(self) -> Callable[[MJCEnvironmentConfig, MJCMorphology], composer.Task]:
        raise NotImplementedError

    def environment(self, robot: MJCRobot, wrap2gym: bool = True) -> Environment:
        return default_make_mjc_env(config=self, robot=robot, wrap2gym=wrap2gym)

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
    def control_timestep(self) -> float:
        return self.num_substeps * self.physics_timestep

    @property
    def physics_timestep(self) -> float:
        return self.original_physics_timestep * self.time_scale

    @property
    def original_physics_timestep(self) -> float:
        return 0.002

    @property
    def camera_ids(self) -> List[int]:
        return [0]
