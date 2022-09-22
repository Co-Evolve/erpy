import math
from pathlib import Path

import gym

from erpy.base.ea import EAConfig
from erpy.base.evaluator import EvaluationCallback
from erpy.base.genome import Genome
from erpy.utils.video import create_video


class VideoCallback(EvaluationCallback):
    def __init__(self, config: EAConfig):
        super().__init__(config, name="VideoCallback")

        self._frames = []
        self._env = None
        self._genome_id = None
        self._episode_index = 0

        self._base_path = Path(self._ea_config.saver_config.analysis_path) / "videos"
        self._base_path.mkdir(parents=True, exist_ok=True)

        max_fps = 60
        max_num_frames = self.config.environment_config.simulation_time * max_fps
        num_frames = self.config.environment_config.num_timesteps
        keep_ratio = max_num_frames / num_frames
        self._keep_every_nth = math.ceil(1 / keep_ratio)
        self._step_index = 0

    def from_env(self, env: gym.Env) -> None:
        self._env = env

    def from_genome(self, genome: Genome) -> None:
        self._genome_id = genome.genome_id

    def before_step(self, observations, actions) -> None:
        if self._step_index % self._keep_every_nth == 0:
            self._frames.append(self._env.render())
        self._step_index += 1

    def after_episode(self) -> None:
        fps = len(self._frames) / self.config.environment_config.simulation_time
        path = self._base_path / f'genome_{self._genome_id}_episode_{self._episode_index}.mp4'
        print(f'Creating video of {len(self._frames)} frames (fps: {fps}) and saving to {str(path)}')
        create_video(frames=self._frames, framerate=fps,
                     out_path=str(path))

        self._episode_index += 1
        self._frames.clear()
        self._step_index = 0
