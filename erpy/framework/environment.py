from __future__ import annotations

import abc
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from erpy.framework import phenome
from erpy.framework.component import EAComponentConfig


@dataclass
class EnvironmentConfig:
    @abc.abstractmethod
    def environment(
            self,
            morphology: phenome.Morphology
            ) -> gym.Env:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def simulation_time(
            self
            ) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_substeps(
            self
            ) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def time_scale(
            self
            ) -> float:
        raise NotImplementedError

    @property
    def original_physics_timestep(
            self
            ) -> float:
        raise NotImplementedError

    @property
    def control_timestep(
            self
            ) -> float:
        return self.num_substeps * self.physics_timestep

    @property
    def physics_timestep(
            self
            ) -> float:
        return self.original_physics_timestep * self.time_scale

    @property
    def num_timesteps(
            self
            ) -> int:
        return int(np.ceil(self.simulation_time / self.control_timestep))
