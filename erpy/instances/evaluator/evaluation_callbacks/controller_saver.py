from pathlib import Path

from erpy.framework.evaluator import EvaluationCallback
from erpy.framework.genome import Genome
from erpy.framework.phenome import Robot


class SaveControllerEvaluationCallback(EvaluationCallback):
    def __init__(self):
        super().__init__()
        self._controller = None
        self._genome_id = None

    def from_genome(self, genome: Genome) -> None:
        self._genome_id = genome.genome_id

    def from_robot(self, robot: Robot) -> None:
        self._controller = robot.controller

    @property
    def output_path(self) -> str:
        base_path = Path(self.ea_config.saver_config.save_path) / "controller_models"
        base_path.mkdir(exist_ok=True, parents=True)
        return str(base_path / f"genome-controller-{self._genome_id}")

    def _save_model(self) -> None:
        path = self.output_path
        self._controller.save(path)

    def after_evaluation(self) -> None:
        self._save_model()
        self._controller = None
        self._genome_id = None
