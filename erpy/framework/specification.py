from __future__ import annotations

import abc
import pickle
from typing import List, Iterable, TYPE_CHECKING, Callable

from erpy.framework.parameters import Parameter, FixedParameter

if TYPE_CHECKING:
    pass


class Specification(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @property
    def parameters(self) -> List[Parameter]:
        parameters = []
        for field_name in vars(self):
            field = self.__getattribute__(field_name)
            if isinstance(field, Parameter):
                parameters.append(field)
            elif isinstance(field, Iterable) and not isinstance(field, str):
                for spec in field:
                    if isinstance(spec, Specification):
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


class RobotSpecification(Specification, metaclass=abc.ABCMeta):
    def __init__(self, morphology_specification: MorphologySpecification,
                 controller_specification: ControllerSpecification | None) -> None:
        super().__init__()
        self._morphology_specification = morphology_specification
        self._controller_specification = controller_specification

    @property
    def morphology_specification(self) -> MorphologySpecification:
        return self._morphology_specification

    @property
    def controller_specification(self) -> ControllerSpecification:
        return self._controller_specification

    @property
    def is_valid(self) -> bool:
        return self.morphology_specification.is_valid and self.controller_specification.is_valid


class MorphologySpecification(Specification, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @property
    def is_valid(self) -> bool:
        return True


class ControllerSpecification(Specification, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @property
    def is_valid(self) -> bool:
        return True


class SpecificationParameterizer(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def parameterize_specification(self, specification: Specification) -> None:
        raise NotImplementedError

    def get_target_parameters(self, specification: Specification) -> List[Parameter]:
        target_parameters = []
        for parameter in specification.parameters:
            if not isinstance(parameter, FixedParameter):
                target_parameters.append(parameter)
        return target_parameters

    def num_target_parameters(self, specification: Specification) -> int:
        target_parameters = self.get_target_parameters(specification)
        num_parameters = len(target_parameters)
        return num_parameters


class RobotSpecificationParameterizer(SpecificationParameterizer, metaclass=abc.ABCMeta):
    def __init__(self,
                 specification_generator: Callable[[], RobotSpecification],
                 morphology_parameterizer: MorphologySpecificationParameterizer,
                 controller_parameterizer: ControllerSpecificationParameterizer) -> None:
        super().__init__()
        self._specification_generator = specification_generator
        self._morphology_parameterizer = morphology_parameterizer
        self._controller_parameterizer = controller_parameterizer

    def parameterize_specification(self, specification: RobotSpecification):
        self._morphology_parameterizer.parameterize_specification(specification.morphology_specification)
        self._controller_parameterizer.parameterize_specification(specification.controller_specification)

    def generate_parameterized_specification(self) -> RobotSpecification:
        robot_specification = self._specification_generator()
        self.parameterize_specification(robot_specification)
        return robot_specification

    def get_target_parameters(self, specification: RobotSpecification) -> List[Parameter]:
        morphology_parameters = self._morphology_parameterizer.get_target_parameters(
            specification.morphology_specification)
        controller_parameters = self._controller_parameterizer.get_target_parameters(
            specification.controller_specification)
        return morphology_parameters + controller_parameters

    def get_parameter_labels(self, specification: RobotSpecification) -> List[str]:
        morphology_labels = self._morphology_parameterizer.get_parameter_labels(specification.morphology_specification)
        controller_labels = self._controller_parameterizer.get_parameter_labels(specification.controller_specification)
        return morphology_labels + controller_labels


class MorphologySpecificationParameterizer(SpecificationParameterizer, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def parameterize_specification(self, specification: MorphologySpecification) -> None:
        raise NotImplementedError

    def get_parameter_labels(self, specification: MorphologySpecification) -> List[str]:
        return []


class ControllerSpecificationParameterizer(SpecificationParameterizer, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def parameterize_specification(self, specification: ControllerSpecification) -> None:
        raise NotImplementedError

    def get_parameter_labels(self, specification: ControllerSpecification) -> List[str]:
        return []
