from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Callable, Iterable

import gymnasium as gym

from erpy.framework.ea import EAConfig
from erpy.framework.evaluator import EvaluatorConfig, Evaluator, EvaluationActor, EvaluationResult
from erpy.framework.genome import Genome
from erpy.framework.phenome import Robot
from erpy.framework.population import Population


@dataclass
class DefaultEvaluatorConfig(EvaluatorConfig):
    reward_aggregator: Callable[[Iterable], float]

    @property
    def evaluator(self) -> Type[DefaultEvaluator]:
        return DefaultEvaluator


class DefaultEvaluator(Evaluator):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(config)
        self._evaluation_actor = DefaultEvaluationActor(config)

    @property
    def config(self) -> DefaultEvaluator:
        return super().config

    def evaluate(self, population: Population) -> None:
        all_genomes = population.genomes
        target_genome_ids = population.to_evaluate

        target_genomes = [all_genomes[genome_id] for genome_id in target_genome_ids]

        evaluation_results = [self._evaluation_actor.evaluate(genome) for genome in target_genomes]

        population.evaluation_results += evaluation_results


class DefaultEvaluationActor(EvaluationActor):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(config)

    @property
    def config(self) -> DefaultEvaluatorConfig:
        return super().config

    def _create_environment(self, robot: Robot) -> gym.Env:
        self._callback.update_environment_config(self.config.environment_config)
        environment = self.config.environment_config.environment(
            morphology=robot.morphology
        )
        return environment

    def evaluate(self, genome: Genome) -> EvaluationResult:
        shared_callback_data = dict()

        self._callback.before_evaluation(
            config=self._ea_config,
            shared_callback_data=shared_callback_data
        )
        self._callback.from_genome(genome)

        robot = self.config.robot(genome.specification)
        self._callback.from_robot(robot)

        env = self._create_environment()
        self._callback.from_env(env)

        self._callback.before_episode()

        observations = env.reset()
        done = False
        rewards = []
        while not done:
            actions = robot(observations)
            self._callback.before_step(
                observations=observations,
                actions=actions
            )

            observations, reward, terminated, truncated, info = env.step(actions)
            rewards.append(reward)

            self._callback.after_step(
                observations=observations,
                actions=actions,
                reward=reward,
                info=info
            )

        env.close()
        self._callback.after_episode()

        fitness = self.config.reward_aggregator(rewards)

        evaluation_result = EvaluationResult(
            genome=genome,
            fitness=fitness
        )
        self._callback.update_evaluation_result(evaluation_result)

        self._callback.after_evaluation()
        return evaluation_result
