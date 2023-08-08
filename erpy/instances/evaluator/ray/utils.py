from typing import Callable

import gym
from stable_baselines3.common.env_util import make_vec_env

from erpy.framework.environment import EnvironmentConfig, Environment
from erpy.framework.phenome import Morphology


def create_vectorized_environment(morphology_generator: Callable[[], Morphology], environment_config: EnvironmentConfig,
                                  number_of_environments: int) -> gym.vector.VectorEnv:
    def make_env() -> Environment:
        morphology = morphology_generator()
        env = environment_config.environment(morphology=morphology)
        return env

    environment = make_vec_env(make_env, n_envs=number_of_environments)
    return environment
