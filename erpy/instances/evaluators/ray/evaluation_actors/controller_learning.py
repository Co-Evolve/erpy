from dataclasses import dataclass
from typing import Type, Callable

import gym.vector
import ray
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from erpy.framework.ea import EAConfig
from erpy.framework.environment import Environment
from erpy.framework.evaluator import EvaluationResult, EvaluationActor
from erpy.framework.genome import Genome
from erpy.framework.phenome import Robot
from erpy.instances.evaluators.ray.evaluator import RayEvaluatorConfig, RayDistributedEvaluator, \
    DistributedEvaluatorConfig


@dataclass
class RayControllerLearningEvaluatorConfig(RayEvaluatorConfig):
    total_timesteps: int = 10000

    @property
    def actor_factory(self) -> Callable[[DistributedEvaluatorConfig], Type[ray.actor.ActorClass]]:
        return ray_controller_learning_evaluation_actor_factory

    @property
    def evaluator(self) -> Type[RayDistributedEvaluator]:
        return RayDistributedEvaluator


def ray_controller_learning_evaluation_actor_factory(config: EAConfig) -> Type[EvaluationActor]:
    @ray.remote(num_cpus=config.evaluator_config.num_cores_per_worker)
    class RayControllerLearningEvaluationActor(EvaluationActor):
        def __init__(self, config: EAConfig) -> None:
            super().__init__(config=config)

        @property
        def config(self) -> RayControllerLearningEvaluatorConfig:
            return super().config

        def _create_environment(self, robot: Robot) -> gym.vector.VectorEnv:
            self._callback.update_environment_config(self.config.environment_config)

            def make_env() -> Environment:
                morphology = self.config.robot(robot.specification).morphology
                env = self.config.environment_config.environment(morphology=morphology)
                return env

            environment = make_vec_env(make_env, n_envs=config.evaluator_config.num_cores_per_worker)
            self._callback.from_env(environment)
            return environment

        def _evaluate_controller(self, robot: Robot) -> float:
            environment = self._create_environment(robot)
            rewards, _ = evaluate_policy(model=robot.controller,
                                         env=environment,
                                         n_eval_episodes=self.config.num_eval_episodes,
                                         return_episode_rewards=True)
            final_reward = self.config.episode_aggregator(rewards)
            environment.close()

            return final_reward

        def _learn_controller(self, robot: Robot) -> None:
            environment = self._create_environment(robot)
            robot.controller.set_environment(environment)
            robot.controller.learn(total_timesteps=self.config.total_timesteps,
                                   callback=self._callback)
            environment.close()

        def evaluate(self, genome: Genome, analyze: bool = False) -> EvaluationResult:
            shared_callback_data = dict()
            self._callback.before_evaluation(config=self._ea_config, shared_callback_data=shared_callback_data)

            self._callback.from_genome(genome)

            robot = self.config.robot(specification=genome.specification)
            self._callback.from_robot(robot=robot)

            # Always set the environment one time first
            dummy_env = self._create_environment(robot=robot)
            robot.controller.set_environment(dummy_env)

            pre_learn_reward = self._evaluate_controller(robot=robot)
            self._learn_controller(robot=robot)
            post_learn_reward = self._evaluate_controller(robot)

            evaluation_result = EvaluationResult(genome=genome, fitness=post_learn_reward,
                                                 info={"episode_failures": {"physics": 0,
                                                                            "validity": 0},
                                                       "rewards": {"pre_learn": pre_learn_reward,
                                                                   "post_learn": post_learn_reward}})
            self._callback.update_evaluation_result(evaluation_result)

            self._callback.after_evaluation()
            return evaluation_result

    return RayControllerLearningEvaluationActor
