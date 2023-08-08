from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Type, Optional, List

import numpy as np

from erpy.framework.genome import GenomeConfig, Genome
from erpy.framework.parameters import ContinuousParameter
from erpy.framework.specification import RobotSpecification
from erpy.utils.math import renormalize


@dataclass
class ESGenomeConfig(GenomeConfig):
    def genome(self) -> Type[ESGenome]:
        raise NotImplementedError

    def rescale_parameters(self, parameters: np.ndarray) -> np.ndarray:
        spec = self.base_specification()
        params = self.extract_parameters(spec)

        rescaled_parameters = []
        for param, value in zip(params, parameters):
            rescaled_value = renormalize(value, [0, 1], [param.low, param.high])
            rescaled_parameters.append(rescaled_value)

        return np.array(rescaled_parameters)

    def normalise_parameters(self, specification: RobotSpecification) -> np.ndarray:
        parameters = self.extract_parameters(specification)

        normalised_parameters = []
        for parameter in parameters:
            normalised_value = renormalize(parameter.value, [parameter.low, parameter.high], [0, 1])
            normalised_parameters.append(normalised_value)

        return np.array(normalised_parameters)

    @property
    def num_parameters(self) -> int:
        return len(self.extract_parameters(self.base_specification()))

    @abc.abstractmethod
    def extract_parameters(self, specification: RobotSpecification) -> List[ContinuousParameter]:
        raise NotImplementedError

    @abc.abstractmethod
    def base_specification(self) -> RobotSpecification:
        raise NotImplementedError



class ESGenome(Genome, abc.ABC):
    def __init__(self, parameters: np.ndarray, config: ESGenomeConfig, genome_id: int,
                 parent_genome_id: Optional[int] = None):
        super().__init__(config, genome_id, parent_genome_id)
        self._parameters = parameters

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @property
    def config(self) -> ESGenomeConfig:
        return self._config

    @staticmethod
    def generate(config: GenomeConfig, genome_id: int, *args, **kwargs) -> ESGenome:
        raise NotImplementedError

    def mutate(self, child_genome_id: int, *args, **kwargs) -> ESGenome:
        raise NotImplementedError

    def cross_over(self, partner_genome: Genome, child_genome_id: int) -> ESGenome:
        raise NotImplementedError

    @property
    def specification(self) -> RobotSpecification:
        if self._specification is None:
            self._specification = self.config.base_specification()
            params = self.config.extract_parameters(specification=self._specification)

            for param, value in zip(params, self._parameters):
                if isinstance(param, ContinuousParameter):
                    value = renormalize(value, [0, 1], [param.low, param.high])
                param.value = value

        return self._specification
