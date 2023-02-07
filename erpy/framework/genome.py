from __future__ import annotations

import abc
import pickle
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Type, List

import numpy as np

from erpy.framework.parameters import ContinuousParameter
from erpy.framework.specification import RobotSpecification
from erpy.utils.math import renormalize


@dataclass
class GenomeConfig(metaclass=abc.ABCMeta):
    random_state: np.random.RandomState

    @property
    @abc.abstractmethod
    def genome(self) -> Type[Genome]:
        raise NotImplementedError


@dataclass
class ESGenomeConfig(GenomeConfig):
    def genome(self) -> Type[ESGenome]:
        raise NotImplementedError

    def rescale_parameters(self, parameters: np.ndarray) -> np.ndarray:
        spec = self.base_specification()
        params = self.extract_parameters(spec)

        rescaled_parameters = []
        for param, value in zip(params, parameters):
            rescaled_value = renormalize(value, [0, 1], [param.low, param.high])
            rescaled_parameters.append(rescaled_value)

        return np.array(rescaled_parameters)

    def normalise_parameters(self, specification: RobotSpecification) -> np.ndarray:
        parameters = self.extract_parameters(specification)

        normalised_parameters = []
        for parameter in parameters:
            normalised_value = renormalize(parameter.value, [parameter.low, parameter.high], [0, 1])
            normalised_parameters.append(normalised_value)

        return np.array(normalised_parameters)

    @property
    def num_parameters(self) -> int:
        return len(self.extract_parameters(self.base_specification()))

    @abc.abstractmethod
    def extract_parameters(self, specification: RobotSpecification) -> List[ContinuousParameter]:
        raise NotImplementedError

    @abc.abstractmethod
    def base_specification(self, ) -> RobotSpecification:
        raise NotImplementedError


class Genome(abc.ABC):
    def __init__(self, config: GenomeConfig, genome_id: int, parent_genome_id: Optional[int] = None) -> None:
        self._config = config
        self._genome_id = genome_id
        self._parent_genome_id = parent_genome_id
        self._specification = None
        self.age = 0

    @property
    def genome_id(self) -> int:
        return self._genome_id

    @genome_id.setter
    def genome_id(self, genome_id: int) -> None:
        self._genome_id = genome_id

    @property
    def parent_genome_id(self) -> int:
        return self._parent_genome_id

    @property
    def config(self) -> GenomeConfig:
        return self._config

    @property
    def specification(self) -> RobotSpecification:
        return self._specification

    @staticmethod
    def generate(config: GenomeConfig, genome_id: int, *args, **kwargs) -> Genome:
        raise NotImplementedError

    def mutate(self, child_genome_id: int, *args, **kwargs) -> Genome:
        raise NotImplementedError

    def cross_over(self, partner_genome: Genome, child_genome_id: int) -> Genome:
        raise NotImplementedError

    def save(self, path: str):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> Genome:
        with open(path, 'rb') as handle:
            return pickle.load(handle)


class DummyGenome(Genome):
    def __init__(self, genome_id: int, specification: RobotSpecification) -> None:
        super(DummyGenome, self).__init__(config=None, genome_id=genome_id, parent_genome_id=None)
        self._specification = specification

    @property
    def specification(self) -> RobotSpecification:
        return self._specification

    @staticmethod
    def generate(config: GenomeConfig, genome_id: int, *args, **kwargs) -> Genome:
        raise NotImplementedError

    def mutate(self, child_genome_id: int, *args, **kwargs) -> Genome:
        raise NotImplementedError

    def cross_over(self, partner_genome: Genome, child_genome_id: int) -> Genome:
        raise NotImplementedError


class ESGenome(Genome, ABC):
    def __init__(self, parameters: np.ndarray, config: ESGenomeConfig, genome_id: int,
                 parent_genome_id: Optional[int] = None):
        super().__init__(config, genome_id, parent_genome_id)
        self._parameters = parameters

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @property
    def config(self) -> ESGenomeConfig:
        return self._config

    @staticmethod
    def generate(config: GenomeConfig, genome_id: int, *args, **kwargs) -> ESGenome:
        raise NotImplementedError

    def mutate(self, child_genome_id: int, *args, **kwargs) -> ESGenome:
        raise NotImplementedError

    def cross_over(self, partner_genome: Genome, child_genome_id: int) -> ESGenome:
        raise NotImplementedError

    @property
    def specification(self) -> RobotSpecification:
        if self._specification is None:
            self._specification = self.config.base_specification()
            params = self.config.extract_parameters(specification=self._specification)

            for param, value in zip(params, self._parameters):
                if isinstance(param, ContinuousParameter):
                    value = renormalize(value, [0, 1], [param.low, param.high])
                param.value = value

        return self._specification
