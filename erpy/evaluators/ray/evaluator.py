from __future__ import annotations

import abc
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Type, cast

from ray.util import ActorPool
from tqdm import tqdm

from erpy.base.evaluator import EvaluatorConfig, EvaluationActor, Evaluator
from erpy.base.population import Population


@dataclass
class DistributedEvaluatorConfig(EvaluatorConfig, ABC):
    num_workers: int
    actor_generator: Callable[[DistributedEvaluatorConfig], Type[EvaluationActor]]
    num_cores_per_worker: int


@dataclass
class RayEvaluatorConfig(DistributedEvaluatorConfig):
    @property
    def evaluator(self) -> Type[RayDistributedEvaluator]:
        return RayDistributedEvaluator


class RayDistributedEvaluator(Evaluator):
    def __init__(self, config: RayEvaluatorConfig) -> None:
        super(RayDistributedEvaluator, self).__init__(config=config)

        self.pool: ActorPool = self._build_pool()

    @property
    def config(self) -> RayEvaluatorConfig:
        return super().config

    def _build_pool(self) -> ActorPool:
        workers = [self.config.actor_generator(self.config).remote(self.config) for _ in range(self.config.num_workers)]
        return ActorPool(workers)

    def evaluate(self, population: Population) -> None:
        all_genomes = population.genomes
        target_genome_ids = population.to_evaluate

        target_genomes = [all_genomes[genome_id] for genome_id in target_genome_ids]

        for genome in tqdm(target_genomes, desc=f"Gen {population.generation}\t-\tSending jobs to workers."):
            self.pool.submit(
                lambda worker, genome: worker.evaluate.remote(genome=genome), genome)
            population.under_evaluation.append(genome.genome_id)

        population.to_evaluate.clear()
        population.evaluation_results.clear()

        timeout = None
        while self.pool.has_next():
            try:
                evaluation_result = self.pool.get_next_unordered(timeout=timeout)
                genome = all_genomes[evaluation_result.genome_id]
                population.evaluation_results.append(evaluation_result)
                population.under_evaluation.remove(genome.genome_id)
                timeout = 5
            except TimeoutError:
                break
