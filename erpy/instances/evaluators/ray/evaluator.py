from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Type, Optional

import ray.actor
from ray.util import ActorPool
from tqdm import tqdm

from erpy.framework.ea import EAConfig
from erpy.framework.evaluator import EvaluatorConfig, EvaluationActor, Evaluator
from erpy.framework.population import Population


@dataclass
class DistributedEvaluatorConfig(EvaluatorConfig, ABC):
    num_workers: int
    num_cores_per_worker: int

    @property
    def actor_factory(self) -> Callable[[DistributedEvaluatorConfig], Type[EvaluationActor]]:
        raise NotImplementedError


@dataclass
class RayEvaluatorConfig(DistributedEvaluatorConfig, ABC):
    evaluation_timeout: Optional[int] = None
    log_to_driver: bool = False
    logging_level: int = logging.ERROR
    debug: bool = False
    cluster: bool = False

    @property
    def actor_factory(self) -> Callable[[DistributedEvaluatorConfig], Type[ray.actor.ActorClass]]:
        raise NotImplementedError

    @property
    def evaluator(self) -> Type[RayDistributedEvaluator]:
        return RayDistributedEvaluator


class RayDistributedEvaluator(Evaluator):
    def __init__(self, config: EAConfig) -> None:
        super(RayDistributedEvaluator, self).__init__(config=config)

        self.pool: ActorPool = None
        self._configure_ray()
        self._build_pool()

    @property
    def config(self) -> RayEvaluatorConfig:
        return super().config

    def _configure_ray(self) -> None:
        ray.init(log_to_driver=self.config.log_to_driver,
                 logging_level=self.config.logging_level,
                 local_mode=self.config.debug,
                 address="auto" if self.config.cluster else None)

    def _build_pool(self) -> None:
        if self.pool is not None:
            del self.pool
        workers = [self.config.actor_factory(self._ea_config).remote(self._ea_config) for _ in
                   range(self.config.num_workers)]
        self.pool = ActorPool(workers)

    def evaluate(self, population: Population) -> None:
        all_genomes = population.genomes
        target_genome_ids = population.to_evaluate

        target_genomes = [all_genomes[genome_id] for genome_id in target_genome_ids]

        for genome in tqdm(target_genomes,
                           desc=f"[RayDistributedEvaluator] Generation {population.generation}\t-\tSending jobs to workers"):
            self.pool.submit(
                lambda worker, genome: worker.evaluate.remote(genome=genome), genome)
            population.under_evaluation.add(genome.genome_id)

        pbar = tqdm(
            desc=f"[RayDistributedEvaluator] Generation {population.generation}\t-\tReceived results from workers",
            total=len(population.under_evaluation))
        timeout = None
        while self.pool.has_next():
            try:
                evaluation_result = self.pool.get_next_unordered(timeout=timeout)
                population.evaluation_results.append(evaluation_result)
                population.under_evaluation.discard(evaluation_result.genome.genome_id)
                timeout = self.config.evaluation_timeout
                pbar.update(1)
            except TimeoutError:
                logging.info("[RayDistributedEvaluator] time threshold exceeded")
                break
        pbar.close()
