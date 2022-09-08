from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Iterable, Type

from base.parameters import Parameter
from base.phenome import Robot, Controller, Morphology


@dataclass
class Specification(metaclass=abc.ABCMeta):
    @property
    def parameters(self) -> List[Parameter]:
        parameters = []
        for field_name in self.__annotations__:
            field = self.__getattribute__(field_name)
            if isinstance(field, Parameter):
                parameters.append(field)
            elif isinstance(field, Iterable):
                for spec in field:
                    if isinstance(field, Specification):
                        parameters += spec.parameters
            elif isinstance(field, Specification):
                parameters += field.parameters
        return parameters


@dataclass
class RobotSpecification(Specification, metaclass=abc.ABCMeta):
    robot: Type[Robot]
    morphology_specification: MorphologySpecification
    controller_specification: ControllerSpecification

    def to_phenome(self) -> Robot:
        return self.robot(self)


@dataclass
class MorphologySpecification(Specification, metaclass=abc.ABCMeta):
    morphology: Type[Morphology]

    def to_phenome(self) -> Morphology:
        return self.morphology(self)


@dataclass
class ControllerSpecification(Specification, metaclass=abc.ABCMeta):
    controller: Type[Controller]

    def to_phenome(self) -> Controller:
        return self.controller(self)
