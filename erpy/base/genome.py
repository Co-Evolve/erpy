from __future__ import annotations

import abc
from typing import Optional

from base.specification import Specification, RobotSpecification, MorphologySpecification, ControllerSpecification


class Genome(metaclass=abc.ABCMeta):
    def __init__(self, genome_id: int, parent_genome_id: Optional[int] = None) -> None:
        self._genome_id = genome_id
        self._parent_genome_id = parent_genome_id
        self.age = 0

    @property
    def genome_id(self) -> int:
        return self._genome_id

    @property
    def parent_genome_id(self) -> int:
        return self._parent_genome_id

    @abc.abstractmethod
    def to_specification(self) -> Specification:
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, cell_path):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def generate(genome_id: int) -> Genome:
        raise NotImplementedError

    @abc.abstractmethod
    def mutate(self, child_genome_id: int) -> Genome:
        raise NotImplementedError

    @abc.abstractmethod
    def cross_over(self, partner_genome: Genome, child_genome_id: int) -> Genome:
        raise NotImplementedError


class RobotGenome(Genome, metaclass=abc.ABCMeta):
    def __init__(self, genome_id: int, parent_genome_id: Optional[int] = None) -> None:
        super().__init__(genome_id, parent_genome_id)

    @abc.abstractmethod
    def to_specification(self) -> RobotSpecification:
        raise NotImplementedError


class MorphologyGenome(Genome, metaclass=abc.ABCMeta):
    def __init__(self, genome_id: int, parent_genome_id: Optional[int] = None) -> None:
        super().__init__(genome_id, parent_genome_id)

    @abc.abstractmethod
    def to_specification(self) -> MorphologySpecification:
        raise NotImplementedError


class ControllerGenome(Genome, metaclass=abc.ABCMeta):
    def __init__(self, genome_id: int, parent_genome_id: Optional[int] = None) -> None:
        super().__init__(genome_id, parent_genome_id)

    @abc.abstractmethod
    def to_specification(self) -> ControllerSpecification:
        raise NotImplementedError
