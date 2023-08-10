from abc import ABC
from dataclasses import dataclass

from erpy.framework.ea import EAConfig


@dataclass
class EAComponentConfig(ABC):
    pass


class EAComponent(ABC):
    def __init__(
            self,
            config: EAConfig
            ) -> None:
        self._ea_config = config

    @property
    def ea_config(
            self
            ) -> EAConfig:
        return self._ea_config

    @property
    def config(
            self
            ) -> EAComponentConfig:
        raise NotImplementedError
