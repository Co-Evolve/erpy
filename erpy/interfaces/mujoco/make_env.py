from typing import Tuple

from dm_control import composer
from stable_baselines3.common.monitor import Monitor

from base.genome import RobotGenome
from base.phenome import Robot
from base.types import Environment
from interfaces.mujoco.gym_wrapper import DMC2GymWrapper
from tasks.task_config import TaskConfig


def make_env(config: TaskConfig, genome: RobotGenome) -> Tuple[Environment, Robot]:
    # Genome to robot
    robot_specification = genome.to_specification()
    robot = robot_specification.to_phenome()

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
    return env, robot
