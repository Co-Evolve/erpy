from __future__ import annotations

import abc
from typing import Optional

from erpy.base.phenome import Phenome, RobotMorphology, RobotController, Robot


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
    def to_phenome(self) -> Phenome:
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, cell_path):
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
    def to_phenome(self) -> Robot:
        raise NotImplementedError


class RobotMorphologyGenome(Genome, metaclass=abc.ABCMeta):
    def __init__(self, genome_id: int, parent_genome_id: Optional[int] = None) -> None:
        super().__init__(genome_id, parent_genome_id)

    @abc.abstractmethod
    def to_phenome(self) -> RobotMorphology:
        raise NotImplementedError


class RobotControllerGenome(Genome, metaclass=abc.ABCMeta):
    def __init__(self, genome_id: int, parent_genome_id: Optional[int] = None) -> None:
        super().__init__(genome_id, parent_genome_id)

    @abc.abstractmethod
    def to_phenome(self) -> RobotController:
        raise NotImplementedError
