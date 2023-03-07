from __future__ import annotations

import abc
import pickle
from typing import Optional, Type

from erpy.framework.specification import RobotSpecification, RobotSpecificationParameterizer


class GenomeConfig(metaclass=abc.ABCMeta):
    def __init__(self, specification_parameterizer: RobotSpecificationParameterizer) -> None:
        self._specification_parameterizer = specification_parameterizer

    @property
    def specification_parameterizer(self) -> RobotSpecificationParameterizer:
        return self._specification_parameterizer

    @property
    @abc.abstractmethod
    def genome(self) -> Type[Genome]:
        raise NotImplementedError


class Genome(abc.ABC):
    def __init__(self, specification: RobotSpecification,
                 config: GenomeConfig, genome_id: int, parent_genome_id: Optional[int] = None) -> None:
        self._specification = specification
        self._config = config
        self._genome_id = genome_id
        self._parent_genome_id = parent_genome_id
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
