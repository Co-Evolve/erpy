from typing import Tuple

from dm_control import composer
from stable_baselines3.common.monitor import Monitor

from brb.tasks.task_config import TaskConfig
from erpy.base.types import Environment
from erpy.interfaces.mujoco.gym_wrapper import DMC2GymWrapper
from erpy.interfaces.mujoco.phenome import MJCRobot


def default_make_mjc_env(config: TaskConfig, robot: MJCRobot) -> Tuple[Environment]:
    task = config.task(config, robot.morphology)
    dm_env = composer.Environment(task=task,
                                  random_state=config.random_state,
                                  time_limit=config.simulation_time)
    env = DMC2GymWrapper(env=dm_env,
                         seed=config.seed,
                         from_pixels=False,
                         camera_ids=[0, 1]
                         )
    env = Monitor(env)
    return env
