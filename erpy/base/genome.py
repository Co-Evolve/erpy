from __future__ import annotations

import abc
import pickle
from dataclasses import dataclass
from typing import Optional, Type

import numpy as np

from base.specification import Specification, RobotSpecification, MorphologySpecification, ControllerSpecification


@dataclass
class GenomeConfig:
    random_state: np.random.RandomState

    @property
    @abc.abstractmethod
    def genome(self) -> Type[Genome]:
        raise NotImplementedError


class Genome(metaclass=abc.ABCMeta):
    def __init__(self, config: GenomeConfig, genome_id: int, parent_genome_id: Optional[int] = None) -> None:
        self._config = config
        self._genome_id = genome_id
        self._parent_genome_id = parent_genome_id
        self.age = 0

    @property
    def genome_id(self) -> int:
        return self._genome_id

    @property
    def parent_genome_id(self) -> int:
        return self._parent_genome_id

    @property
    def config(self) -> GenomeConfig:
        return self._config

    @abc.abstractmethod
    def to_specification(self) -> Specification:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def generate(config: GenomeConfig, genome_id: int) -> Genome:
        raise NotImplementedError

    def mutate(self, child_genome_id: int) -> Genome:
        raise NotImplementedError

    def cross_over(self, partner_genome: Genome, child_genome_id: int) -> Genome:
        raise NotImplementedError

    def save(self, path: str):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


class RobotGenome(Genome, metaclass=abc.ABCMeta):
    def __init__(self, config: GenomeConfig, morphology_genome: MorphologyGenome, controller_genome: ControllerGenome,
                 genome_id: int, parent_genome_id: Optional[int] = None) -> None:
        super().__init__(config, genome_id, parent_genome_id)
        self._morphology_genome = morphology_genome
        self._controller_genome = controller_genome

    @property
    def morphology_genome(self) -> MorphologyGenome:
        return self._morphology_genome

    @property
    def controller_genome(self) -> ControllerGenome:
        return self._controller_genome

    @abc.abstractmethod
    def to_specification(self) -> RobotSpecification:
        raise NotImplementedError


class MorphologyGenome(Genome, metaclass=abc.ABCMeta):
    def __init__(self, config: GenomeConfig, genome_id: int, parent_genome_id: Optional[int] = None) -> None:
        super().__init__(config=config, genome_id=genome_id, parent_genome_id=parent_genome_id)

    @abc.abstractmethod
    def to_specification(self) -> MorphologySpecification:
        raise NotImplementedError


class ControllerGenome(Genome, metaclass=abc.ABCMeta):
    def __init__(self, config: GenomeConfig, genome_id: int, parent_genome_id: Optional[int] = None) -> None:
        super().__init__(config=config, genome_id=genome_id, parent_genome_id=parent_genome_id)

    @abc.abstractmethod
    def to_specification(self) -> ControllerSpecification:
        raise NotImplementedError
