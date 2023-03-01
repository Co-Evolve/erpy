import wandb

from erpy.framework.evaluator import EvaluationCallback
from erpy.framework.genome import Genome
from erpy.instances.evaluators.ray.evaluation_actors.controller_learning import RayControllerLearningEvaluatorConfig
from erpy.instances.loggers.wandb_logger import WandBLoggerConfig


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

    def _initialise_wandb(self, genome_id) -> None:
        assert isinstance(self.ea_config.logger_config,
                          WandBLoggerConfig), "WandbEvaluationCallback requires the Logger to be" \
                                              " a (subtype of) WandBLogger"
        project_name = self.logger_config.project_name
        group = self.logger_config.group
        tags = self.logger_config.tags
        name = f"{self.logger_config.run_name}-genome-{genome_id}"
        self._wandb_run = wandb.init(project=project_name, group=group, tags=tags, name=name,
                                     resume="allow", id=name,
                                     sync_tensorboard=True)

    def from_genome(self, genome: Genome) -> None:
        self._initialise_wandb(genome.genome_id)

    def after_evaluation(self) -> None:
        self._wandb_run.finish()
