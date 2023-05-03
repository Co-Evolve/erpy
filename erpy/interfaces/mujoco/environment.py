from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, List, Dict

import dm_env.specs
import gymnasium as gym
from dm_control import composer

import erpy
from erpy.framework.environment import EnvironmentConfig, Environment
from erpy.interfaces.mujoco.gym_wrapper import DMC2GymWrapper
from erpy.interfaces.mujoco.phenome import MJCMorphology


def default_make_mjc_env(config: MJCEnvironmentConfig, morphology: MJCMorphology) -> composer.Environment:
    task = config.task(config, morphology)
    task.set_timesteps(control_timestep=config.control_timestep,
                       physics_timestep=config.physics_timestep)
    env = composer.Environment(task=task,
                               random_state=erpy.random_state,
                               time_limit=config.simulation_time)
    return env


def dm_control_to_gym_environment(config: MJCEnvironmentConfig, environment: composer.Environment) -> gym.Env:
    env = DMC2GymWrapper(env=environment,
                         seed=erpy.seed,
                         from_pixels=False,
                         camera_ids=config.camera_ids)
    return env


@dataclass
class MJCEnvironmentConfig(EnvironmentConfig, abc.ABC):
    @property
    @abc.abstractmethod
    def task(self) -> Callable[[MJCEnvironmentConfig, MJCMorphology], composer.Task]:
        raise NotImplementedError

    def environment(self, morphology: MJCMorphology, wrap2gym: bool = True) -> Environment:
        env = default_make_mjc_env(config=self, morphology=morphology)
        self._observation_specification = env.observation_spec()
        self._action_specification = env.action_spec()
        if wrap2gym:
            env = dm_control_to_gym_environment(config=self, environment=env)
        return env

    @property
    def observation_specification(self) -> Dict[str, dm_env.specs.Array]:
        return self._observation_specification

    @property
    def action_specification(self) -> dm_env.specs.Array:
        return self._action_specification

    @property
    def original_physics_timestep(self) -> float:
        return 0.002

    @property
    def camera_ids(self) -> List[int]:
        return [0]

    @property
    def time_scale(self) -> float:
        return 1.0
