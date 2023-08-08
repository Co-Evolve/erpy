import numpy as np
from dm_control import viewer
from dm_env import TimeStep

from erpy.interfaces.mujoco.environment import MJCEnvironmentConfig, dm_control_to_gym_environment
from erpy.interfaces.mujoco.gym_wrapper import extract_dict_observations_from_dm_control_timestep
from erpy.interfaces.mujoco.phenome import MJCRobot


def evaluate_with_dm_control_viewer(
        env_config: MJCEnvironmentConfig,
        robot: MJCRobot
        ) -> None:
    dm_env = env_config.environment(
            morphology=robot.morphology, wrap2gym=False
            )
    gym_env = dm_control_to_gym_environment(config=env_config, environment=dm_env)

    def policy_fn(
            timestep: TimeStep
            ) -> np.ndarray:
        observations = extract_dict_observations_from_dm_control_timestep(
                timestep=timestep, observation_space=gym_env.observation_space
                )
        actions = robot(observations)
        return actions

    viewer.launch(environment_loader=dm_env, policy=policy_fn)
