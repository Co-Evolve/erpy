import logging
from pathlib import Path

import gym
import numpy as np
from PIL import Image

from erpy.framework.evaluator import EvaluationCallback
from erpy.framework.genome import Genome
from erpy.utils.video import create_video


class VideoCallback(EvaluationCallback):
    def __init__(
            self
            ):
        super().__init__()
        self._frames = []
        self._env = None
        self._genome_id = None
        self._episode_index = 0

        self._save_path = None

    def from_env(
            self,
            env: gym.Env
            ) -> None:
        self._env = env

    def from_genome(
            self,
            genome: Genome
            ) -> None:
        self._genome_id = genome.genome_id

    def before_step(
            self,
            observations,
            actions
            ) -> None:
        self._frames.append(self._env.render())

    def before_episode(
            self
            ) -> None:
        self._save_path = Path(self._ea_config.saver_config.analysis_path) / "videos" / f"episode_{self._episode_index}"
        self._save_path.mkdir(parents=True, exist_ok=True)

    def after_episode(
            self
            ) -> None:
        fps = len(self._frames) / self.config.environment_config.simulation_time
        path = self._save_path / f'genome_{self._genome_id}_episode_{self._episode_index}.mp4'
        logging.info(f'Creating video of {len(self._frames)} frames (fps: {fps}) and saving to {str(path)}')

        create_video(
            frames=self._frames, framerate=fps, out_path=str(path)
            )

        self._episode_index += 1
        self._frames.clear()


class FrameSaverCallback(EvaluationCallback):
    def __init__(
            self,
            save_frequency: int = 1
            ):
        super().__init__()

        self._frames = []
        self._env = None
        self._genome_id = None
        self._episode_index = 0

        self._step_index = 0
        self._save_path = None
        self._save_frequency = save_frequency

    def from_env(
            self,
            env: gym.Env
            ) -> None:
        self._env = env

    def from_genome(
            self,
            genome: Genome
            ) -> None:
        self._genome_id = genome.genome_id

    def before_step(
            self,
            observations,
            actions
            ) -> None:
        if self._step_index % self._save_frequency == 0:
            self._frames.append(self._env.render())
        self._step_index += 1

    def before_episode(
            self
            ) -> None:
        self._save_path = Path(self._ea_config.saver_config.analysis_path) / "frames" / f"episode_{self._episode_index}"
        self._save_path.mkdir(parents=True, exist_ok=True)

    def after_episode(
            self
            ) -> None:
        for i, frame in enumerate(self._frames):
            timestamp = i * self.config.environment_config.control_timestep
            path = self._save_path / f'genome_{self._genome_id}_frame_{timestamp}_s.png'
            frame = np.flip(frame, axis=2)
            image = Image.fromarray(frame, mode="RGB")
            image.save(path)

        self._episode_index += 1
        self._frames.clear()
