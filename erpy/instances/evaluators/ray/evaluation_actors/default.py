from dataclasses import dataclass
from typing import Type, Callable

import ray
from dm_control.rl.control import PhysicsError

from erpy.framework.ea import EAConfig
from erpy.framework.evaluator import EvaluationResult, EvaluationActor
from erpy.framework.genome import Genome
from erpy.instances.evaluators.ray.evaluator import DistributedEvaluatorConfig, RayEvaluatorConfig, \
    RayDistributedEvaluator


@dataclass
class RayDefaultEvaluatorConfig(RayEvaluatorConfig):
    @property
    def actor_factory(self) -> Callable[[DistributedEvaluatorConfig], Type[ray.actor.ActorClass]]:
        return ray_default_evaluation_actor_factory

    @property
    def evaluator(self) -> Type[RayDistributedEvaluator]:
        return RayDistributedEvaluator


def ray_default_evaluation_actor_factory(config: EAConfig) -> Type[EvaluationActor]:
    @ray.remote(num_cpus=config.evaluator_config.num_cores_per_worker)
    class RayDefaultEvaluationActor(EvaluationActor):
        def __init__(self, config: EAConfig) -> None:
            super().__init__(config=config)

        @property
        def config(self) -> RayDefaultEvaluatorConfig:
            return super().config

        def evaluate(self, genome: Genome) -> EvaluationResult:
            shared_callback_data = dict()
            self._callback.before_evaluation(config=self._ea_config, shared_callback_data=shared_callback_data)

            self._callback.from_genome(genome)

            robot, env = None, None

            episode_fitnesses = []
            episode_frames = []
            physics_failures = 0
            validity_failures = 0
            for episode in range(self.config.num_eval_episodes):
                try:
                    self._callback.before_episode()

                    if robot is None or self.config.hard_episode_reset:
                        specification = genome.specification
                        if not specification.is_valid:
                            raise AssertionError

                        robot = self.config.robot(genome.specification)
                        self._callback.from_robot(robot)

                        self._callback.update_environment_config(self.config.environment_config)
                        env = self.config.environment_config.environment(morphology=robot.morphology)
                        robot.controller.set_environment(env)
                        self._callback.from_env(env)

                    robot.reset()
                    observations = env.reset()

                    frames = []
                    rewards = []

                    done = False
                    while not done:
                        actions = robot(observations)

                        self._callback.before_step(observations=observations, actions=actions)
                        observations, reward, done, info = env.step(actions)

                        self._callback.after_step(observations=observations, actions=actions, info=info)

                        rewards.append(reward)

                    self._callback.after_episode()

                    episode_fitness = self.config.reward_aggregator(rewards)
                    episode_fitnesses.append(episode_fitness)

                    episode_frames.append(frames)
                except PhysicsError:
                    physics_failures += 1
                    env.close()
                    episode_fitnesses.append(-1)
                    break
                except AssertionError:
                    validity_failures += 1
                    env.close()
                    episode_fitnesses.append(-1)

            env.close()
            fitness = self.config.episode_aggregator(episode_fitnesses)
            evaluation_result = EvaluationResult(genome=genome, fitness=fitness,
                                                 info={"episode_failures": {"physics": physics_failures,
                                                                            "validity": validity_failures}})
            self._callback.update_evaluation_result(evaluation_result)

            self._callback.after_evaluation()
            return evaluation_result

    return RayDefaultEvaluationActor
