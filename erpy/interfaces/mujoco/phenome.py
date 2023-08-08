from __future__ import annotations

import abc
from abc import ABC
from typing import List, Union

import numpy as np
from dm_control import composer, mjcf
from dm_control.mjcf import export_with_assets
from scipy.spatial.transform import Rotation

from erpy.framework.phenome import Morphology, Robot
from erpy.framework.specification import ControllerSpecification, MorphologySpecification, RobotSpecification


class MJCRobot(Robot, ABC):
    def __init__(
            self,
            specification: RobotSpecification
            ) -> None:
        super().__init__(specification)

    @property
    def morphology(
            self
            ) -> MJCMorphology:
        return super().morphology


class MJCMorphology(Morphology, composer.Entity, metaclass=abc.ABCMeta):
    def __init__(
            self,
            specification: RobotSpecification
            ) -> None:
        self._mjcf_model = mjcf.RootElement(model="morphology")
        self._mjcf_body = self._mjcf_model.worldbody.add('body')
        Morphology.__init__(self, specification)
        composer.Entity.__init__(self)

    @property
    def actuators(
            self
            ) -> List[mjcf.Element]:
        return self.mjcf_model.find_all('actuator')

    @property
    def sensors(
            self
            ) -> List[mjcf.Element]:
        return self.mjcf_model.find_all('sensor')

    @property
    def mjcf_body(
            self
            ) -> mjcf.Element:
        return self._mjcf_body

    @property
    def mjcf_model(
            self
            ) -> mjcf.RootElement:
        return self._mjcf_model

    def after_attachment(
            self
            ) -> None:
        # Called when the morphology is added to the environment
        pass

    @property
    def world_coordinates(
            self
            ) -> np.ndarray:
        return np.zeros(3)

    @property
    def coordinate_frame_in_world(
            self
            ) -> (np.ndarray, np.ndarray):
        return np.zeros(3), Rotation.from_euler('xyz', [0, 0, 0]).as_matrix()

    def world_coordinates_of_point(
            self,
            point: np.ndarray
            ) -> np.ndarray:
        return point

    def export_to_xml_with_assets(
            self,
            output_directory: str = "./mjcf"
            ) -> None:
        export_with_assets(
                mjcf_model=self.mjcf_model, out_dir=output_directory
                )


class MJCMorphologyPart(Morphology, metaclass=abc.ABCMeta):
    def __init__(
            self,
            parent: Union[MJCMorphology, MJCMorphologyPart],
            name: str,
            pos: np.array,
            euler: np.array,
            *args,
            **kwargs
            ) -> None:
        self._parent = parent
        self._name = name
        self._mjcf_body = parent.mjcf_body.add(
                'body', name=self._name, pos=pos, euler=euler
                )
        super().__init__(self.specification)
        self._coordinate_frame_in_world = None
        self._build(*args, **kwargs)

    @property
    def specification(
            self
            ) -> RobotSpecification:
        return self._parent.specification

    @property
    def morphology_specification(
            self
            ) -> MorphologySpecification:
        return self._parent.morphology_specification

    @property
    def controller_specification(
            self
            ) -> ControllerSpecification:
        return self._parent.controller_specification

    @property
    def mjcf_model(
            self
            ) -> mjcf.RootElement:
        return self._parent.mjcf_model

    @property
    def mjcf_body(
            self
            ) -> mjcf.Element:
        return self._mjcf_body

    @property
    def base_name(
            self
            ) -> str:
        return self._name

    @abc.abstractmethod
    def _build(
            self,
            *args,
            **kwargs
            ) -> None:
        raise NotImplementedError

    @property
    def coordinate_frame_in_world(
            self
            ) -> (np.ndarray, np.ndarray):
        """
        Returns the object's coordinate frame with respect to the morphology's world frame:
            0 -> the object's coordinates in world frame
            1 -> a rotation matrix that represents the transformation from the object's local frame to the world frame
        :return:
        """
        if self._coordinate_frame_in_world is None:
            parent_origin, parent_rot = self._parent.coordinate_frame_in_world

            pos = self.mjcf_body.pos
            euler = self.mjcf_body.euler
            my_rot = Rotation.from_euler('xyz', euler).as_matrix()

            self._coordinate_frame_in_world = parent_origin + parent_rot.dot(pos), parent_rot.dot(my_rot)
        return self._coordinate_frame_in_world

    def world_coordinates_of_point(
            self,
            point: np.ndarray
            ) -> np.ndarray:
        """ Returns the world coordinates of the given point in local coordinates"""
        origin, rot = self.coordinate_frame_in_world
        return origin + rot.dot(point)

    @property
    def world_coordinates(
            self
            ) -> np.ndarray:
        return self.coordinate_frame_in_world[0]
