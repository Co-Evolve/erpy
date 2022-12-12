from typing import Type

import ray
from dm_control.rl.control import PhysicsError

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

            robot, env = None, None

            episode_fitnesses = []
            episode_frames = []
            physics_failures = 0
            for episode in range(self.config.num_eval_episodes):
                try:
                    self.callback_handler.before_episode()

                    if robot is None or self.config.hard_episode_reset:
                        robot = self.config.robot(genome.specification)
                        self.callback_handler.from_robot(robot)

                        self.callback_handler.update_environment_config(self.config.environment_config)
                        env = self.config.environment_config.environment(robot=robot)
                        self.callback_handler.from_env(env)

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
                except PhysicsError:
                    physics_failures += 1
                    env.close()
                    episode_fitnesses.append(-1)
                    break

            env.close()
            fitness = self.config.episode_aggregator(episode_fitnesses)
            evaluation_result = EvaluationResult(genome_id=genome.genome_id, fitness=fitness,
                                                 info={"episode_failures": {"physics": physics_failures}})
            evaluation_result = self.callback_handler.update_evaluation_result(evaluation_result)

            return evaluation_result

    return BaseEvaluationActor
