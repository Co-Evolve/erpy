from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Union

import dm_control.composer
import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

import erpy.framework.phenome as phenome

Environment = Union[gym.Env, dm_control.composer.Environment, VecEnv]


@dataclass
class EnvironmentConfig(metaclass=abc.ABCMeta):
    _observation_specification = None
    _action_specification = None

    @abc.abstractmethod
    def environment(self, morphology: phenome.Morphology) -> Environment:
        raise NotImplementedError

    def observation_specification(self) -> gym.Space:
        return self._observation_specification

    def action_specification(self) -> gym.Space:
        return self._action_specification

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
        raise NotImplementedError

    @property
    def num_timesteps(self) -> int:
        return int(np.ceil(self.simulation_time / self.control_timestep))
