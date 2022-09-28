from __future__ import annotations

import abc
import pickle
from dataclasses import dataclass
from typing import List, Iterable

from erpy.base.parameters import Parameter


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

    def save(self, path: str):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> Specification:
        with open(path, 'rb') as handle:
            return pickle.load(handle)


@dataclass
class RobotSpecification(Specification, metaclass=abc.ABCMeta):
    morphology_specification: MorphologySpecification
    controller_specification: ControllerSpecification


@dataclass
class MorphologySpecification(Specification, metaclass=abc.ABCMeta):
    pass


@dataclass
class ControllerSpecification(Specification, metaclass=abc.ABCMeta):
    pass
