import numpy as np
from dm_control import viewer
from dm_env import TimeStep

from erpy.interfaces.mujoco.environment import MJCEnvironmentConfig
from erpy.interfaces.mujoco.gym_wrapper import get_clean_obs, vectorize_observations
from erpy.interfaces.mujoco.phenome import MJCRobot


def evaluate_with_dm_control_viewer(env_config: MJCEnvironmentConfig, robot: MJCRobot) -> None:
    dm_env = env_config.environment(morphology=robot.morphology,
                                    wrap2gym=False)

    def policy_fn(timestep: TimeStep) -> np.ndarray:
        observations = get_clean_obs(timestep)
        observations = vectorize_observations(observations)
        actions = robot(observations)[0]
        return actions

    viewer.launch(dm_env, policy=policy_fn)
