import numpy as np
from dm_control import viewer
from dm_env import TimeStep

from erpy.interfaces.mujoco.environment import MJCEnvironmentConfig
from erpy.interfaces.mujoco.gym_wrapper import _flatten_obs, get_clean_obs
from erpy.interfaces.mujoco.phenome import MJCRobot


def evaluate_with_dm_control_viewer(env_config: MJCEnvironmentConfig, robot: MJCRobot) -> None:
    dm_env = env_config.environment(morphology=robot.morphology,
                                    wrap2gym=False)

    def policy_fn(timestep: TimeStep) -> np.ndarray:
        observations = get_clean_obs(timestep)
        actions = robot(observations)
        return actions

    viewer.launch(dm_env, policy=policy_fn)
