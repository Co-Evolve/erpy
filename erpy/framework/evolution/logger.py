from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

from erpy.framework.component import EAComponent, EAComponentConfig
from erpy.framework.population import Population

if TYPE_CHECKING:
    from erpy.framework.ea import EAConfig


@dataclass
class LoggerConfig(EAComponentConfig):
    @property
    @abc.abstractmethod
    def logger(
            self
            ) -> Type[Logger]:
        raise NotImplementedError


class Logger(EAComponent):
    def __init__(
            self,
            config: EAConfig
            ) -> None:
        super().__init__(config)

    @property
    def config(
            self
            ) -> LoggerConfig:
        return self.ea_config.logger_config

    @abc.abstractmethod
    def log(
            self,
            population: Population
            ) -> None:
        raise NotImplementedError
