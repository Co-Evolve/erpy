from __future__ import annotations

import abc
from typing import Tuple, Union

import numpy as np
from dm_control import composer, mjcf

from base.phenome import Morphology, Robot
from base.specification import MorphologySpecification, RobotSpecification


class MJCRobot(Robot, metaclass=abc.ABCMeta):
    def __init__(self, specification: RobotSpecification):
        super().__init__(specification)

    @property
    def morphology(self) -> MJCMorphology:
        return self.morphology


class MJCMorphology(Morphology, composer.Entity, metaclass=abc.ABCMeta):
    def __init__(self, specification: MorphologySpecification) -> None:
        self._mjcf_model = mjcf.RootElement()
        super().__init__(specification=specification)

    @property
    def specification(self) -> MorphologySpecification:
        return self._parent.specification

    @property
    def actuators(self) -> Tuple[mjcf.Element]:
        return tuple(self.mjcf_model.find_all('actuator'))

    @property
    def mjcf_body(self) -> mjcf.Element:
        return self.mjcf_model.worldbody

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._mjcf_model


class MJCMorphologyPart(Morphology, metaclass=abc.ABCMeta):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart],
                 name: str,
                 pos: np.array,
                 euler: np.array,
                 *args, **kwargs) -> None:
        super().__init__(self.specification)
        self._parent = parent
        self._mjcf_body = parent.mjcf_body.add('body',
                                               name=name,
                                               pos=pos,
                                               euler=euler)
        self._build(*args, **kwargs)

    @property
    def specification(self) -> MorphologySpecification:
        return self._parent.specification

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._parent.mjcf_model

    @property
    def mjcf_body(self) -> mjcf.Element:
        return self._mjcf_body

    @abc.abstractmethod
    def _build(self, *args, **kwargs) -> None:
        raise NotImplementedError
