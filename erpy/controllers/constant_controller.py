from dataclasses import dataclass

import numpy as np

from erpy.base.parameters import FixedParameter, ContinuousParameter
from erpy.base.phenome import Controller
from erpy.base.specification import ControllerSpecification


@dataclass
class ConstantControllerSpecification(ControllerSpecification):
    _n_outputs: FixedParameter
    _constant: ContinuousParameter

    @property
    def n_outputs(self) -> int:
        return self._n_outputs.value

    @property
    def constant(self) -> float:
        return self._constant.value


class ConstantController(Controller):
    def __init__(self, specification: ConstantControllerSpecification):
        super().__init__(specification)

    @property
    def specification(self) -> ConstantControllerSpecification:
        return super().specification

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return np.ones(self.specification.n_outputs) * self.specification.constant
