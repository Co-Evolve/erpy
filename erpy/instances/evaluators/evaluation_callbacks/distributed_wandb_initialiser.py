from typing import Dict, Any

from erpy.framework.ea import EAConfig
from erpy.framework.evaluator import EvaluationCallback
from erpy.framework.genome import Genome
from erpy.instances.evaluators.ray.evaluation_actors.controller_learning import RayControllerLearningEvaluatorConfig
from erpy.instances.loggers.wandb_logger import wandb_initialise_run, WandBLoggerConfig


class DistributedWandbInitialisationEvaluationCallback(EvaluationCallback):
    def __init__(self) -> None:
        super().__init__()

        self._wandb_run = None

    @property
    def config(self) -> RayControllerLearningEvaluatorConfig:
        return super().config

    @property
    def logger_config(self) -> WandBLoggerConfig:
        return self.ea_config.logger_config

    def before_evaluation(self, config: EAConfig, shared_callback_data: Dict[str, Any]) -> None:
        super().before_evaluation(config=config, shared_callback_data=shared_callback_data)
        assert isinstance(config.logger_config, WandBLoggerConfig), "WandbEvaluationCallback requires the Logger to be" \
                                                                    " a (subtype of) WandBLogger"
        project_name = self.logger_config.project_name
        group = self.logger_config.group
        tags = self.logger_config.tags
        self._wandb_run = wandb_initialise_run(project=project_name, group=group, tags=tags)

    def from_genome(self, genome: Genome) -> None:
        run_name = self.logger_config.run_name
        self._wandb_run.name = f"{run_name}-genome-{genome.genome_id}"

    def after_evaluation(self) -> None:
        self._wandb_run.finish()
