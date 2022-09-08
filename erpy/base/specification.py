import abc
from dataclasses import dataclass
from typing import List, Iterable

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
    @abc.abstractmethod
    def to_phenome(self) -> Robot:
        pass


@dataclass
class MorphologySpecification(Specification, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_phenome(self) -> Morphology:
        pass


@dataclass
class ControllerSpecification(Specification, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def to_phenome(self) -> Controller:
        pass
