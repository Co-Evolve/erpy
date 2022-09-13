from __future__ import annotations

import abc
from dataclasses import dataclass

import gym
import numpy as np

from erpy.base.phenome import Robot

Environment = gym.Env


@dataclass
class EnvironmentConfig(metaclass=abc.ABCMeta):
    seed: int
    random_state: np.random.RandomState

    @abc.abstractmethod
    def environment(self, robot: Robot) -> Environment:
        raise NotImplementedError
