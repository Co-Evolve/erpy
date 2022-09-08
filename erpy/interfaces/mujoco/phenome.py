import abc

from dm_control import composer
from torch import Type

from base.genome import RobotMorphologyGenome
from base.phenome import RobotMorphology
from creatures.creature import CreatureMorphologySpecification


class MJCRobotMorphology(RobotMorphology, composer.Entity, metaclass=abc.ABCMeta):
    def __init__(self, genome: RobotMorphologyGenome, observables: Type[composer.Observables]):
        self._morphology_spec = genome.to_phenome()
        self._observables = observables
        super().__init__(genome=genome)

    @property
    def morphology_spec(self) -> CreatureMorphologySpecification:
        return self._morphology_spec
