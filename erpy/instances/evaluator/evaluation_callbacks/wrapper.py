from typing import Type, Dict, Any

from stable_baselines3.common.callbacks import BaseCallback

from erpy.framework.ea import EAConfig
from erpy.framework.evaluator import EvaluationCallback
from erpy.framework.genome import Genome


class EvaluationCallbackWrapper(EvaluationCallback):
    def __init__(self, callback: Type[BaseCallback], **kwargs):
        super().__init__()
        self._callback_creator = callback
        self._callback = None
        self._kwargs = kwargs

    def before_evaluation(self, config: EAConfig, shared_callback_data: Dict[str, Any]) -> None:
        super().before_evaluation(config=config, shared_callback_data=shared_callback_data)

    def from_genome(self, genome: Genome) -> None:
        self._callback = self._callback_creator(**self._kwargs)

    def __getattr__(self, item):
        if item == "_callback":
            return self.__getattribute__(item)
        try:
            return self._callback.__getattr__(item)
        except AttributeError:
            return self.__getattribute__(item)

    def after_evaluation(self) -> None:
        self._callback = None
