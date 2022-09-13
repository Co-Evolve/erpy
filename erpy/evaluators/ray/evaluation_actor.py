from typing import Type

import ray

from erpy.base.ea import EAConfig
from erpy.base.evaluator import EvaluatorConfig, EvaluationResult, EvaluationActor
from erpy.base.genome import RobotGenome
from erpy.evaluators.ray.evaluator import DistributedEvaluatorConfig
from erpy.utils.video import create_video


def make_base_evaluation_actor(config: DistributedEvaluatorConfig) -> Type[EvaluationActor]:
    @ray.remote(num_cpus=config.num_cores_per_worker)
    class BaseEvaluationActor(EvaluationActor):
        def __init__(self, config: EAConfig) -> None:
            super().__init__(config=config)

        @property
        def config(self) -> EvaluatorConfig:
            return super().config

        def evaluate(self, genome: RobotGenome) -> EvaluationResult:
            self.callback_handler.reset()

            robot = self.config.robot(genome.specification)
            env = self.config.environment_config.environment(robot=robot)

            self.callback_handler.from_genome(genome)
            self.callback_handler.from_robot(robot)

            episode_fitnesses = []
            episode_frames = []
            for episode in range(self.config.num_eval_episodes):
                self.callback_handler.before_episode()

                robot.reset()
                observations = env.reset()

                frames = []
                rewards = []

                done = False
                while not done:
                    if self.config.render:
                        frames.append(env.render())

                    actions = robot(observations)

                    self.callback_handler.before_step(observations=observations, actions=actions)
                    observations, reward, done, _ = env.step(actions)
                    self.callback_handler.after_step(observations=observations, actions=actions)

                    rewards.append(reward)
                env.close()

                self.callback_handler.after_episode()

                if self.config.render:
                    create_video(frames=frames, framerate=1. / env.env._env.control_timestep(),
                                 out_path=f'genome_{genome.genome_id}.mp4')

                episode_fitness = self.config.reward_aggregator(rewards)
                episode_fitnesses.append(episode_fitness)

                episode_frames.append(frames)

            fitness = self.config.episode_aggregator(episode_fitnesses)
            evaluation_result = EvaluationResult(genome_id=genome.genome_id, fitness=fitness, info=dict())
            evaluation_result = self.callback_handler.update_evaluation_result(evaluation_result)

            return evaluation_result

    return BaseEvaluationActor
