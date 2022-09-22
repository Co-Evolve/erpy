from typing import Type

import ray

from erpy.base.ea import EAConfig
from erpy.base.evaluator import EvaluatorConfig, EvaluationResult, EvaluationActor
from erpy.base.genome import Genome


def make_base_evaluation_actor(config: EAConfig) -> Type[EvaluationActor]:
    @ray.remote(num_cpus=config.evaluator_config.num_cores_per_worker)
    class BaseEvaluationActor(EvaluationActor):
        def __init__(self, config: EvaluatorConfig) -> None:
            super().__init__(config=config)

        @property
        def config(self) -> EvaluatorConfig:
            return super().config

        def evaluate(self, genome: Genome, analyze: bool = False) -> EvaluationResult:
            self.callback_handler.reset(analyze=analyze)
            self.callback_handler.from_genome(genome)

            robot = self.config.robot(genome.specification)
            self.callback_handler.from_robot(robot)

            env = self.config.environment_config.environment(robot=robot)
            self.callback_handler.from_env(env)

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
                    actions = robot(observations)

                    self.callback_handler.before_step(observations=observations, actions=actions)
                    observations, reward, done, info = env.step(actions)

                    self.callback_handler.after_step(observations=observations, actions=actions, info=info)

                    rewards.append(reward)

                self.callback_handler.after_episode()

                episode_fitness = self.config.reward_aggregator(rewards)
                episode_fitnesses.append(episode_fitness)

                episode_frames.append(frames)

            env.close()
            fitness = self.config.episode_aggregator(episode_fitnesses)
            evaluation_result = EvaluationResult(genome_id=genome.genome_id, fitness=fitness, info=dict())
            evaluation_result = self.callback_handler.update_evaluation_result(evaluation_result)

            return evaluation_result

    return BaseEvaluationActor
