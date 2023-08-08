from typing import Any, Dict, List

import dm_control.composer
import gymnasium as gym
import numpy as np
from dm_control.mujoco.wrapper import MjvOption
from dm_env import TimeStep, specs
from gymnasium import spaces
from gymnasium.core import ObsType
from numpy import number

from erpy import seed as erpy_seed


def _extract_bounds_from_dm_control_spec(
        spec: specs.Array | specs.BoundedArray
        ) -> (np.ndarray, np.ndarray):
    assert spec.dtype == np.float64 or spec.dtype == np.float32
    dim = int(np.prod(spec.shape))
    if type(spec) == specs.Array:
        bound = np.inf * np.ones(dim, dtype=np.float32)
        return -bound, bound
    elif type(spec) == specs.BoundedArray:
        zeros = np.zeros(dim, dtype=np.float32)
        return spec.minimum + zeros, spec.maximum + zeros
    else:
        raise TypeError(f"Given action specification type not supported: {type(spec)}")


def _dm_control_array_spec_to_gym_box(
        spec: specs.Array | specs.BoundedArray,
        dtype
        ) -> spaces.Box:

    lower_bound, upper_bound = _extract_bounds_from_dm_control_spec(spec)
    return spaces.Box(low=lower_bound, high=upper_bound, dtype=dtype)


def _dm_control_dict_spec_to_gym_dict(
        spec: Dict[str, specs.Array | specs.BoundedArray],
        dtype: number
        ) -> spaces.Dict:
    gym_dict = dict()
    for key, sub_spec in spec.items():
        gym_dict[key] = _dm_control_spec_to_gym_space(spec=sub_spec, dtype=dtype)
    return spaces.Dict(gym_dict)


def _dm_control_spec_to_gym_space(
        spec: specs.Array | Dict[str, specs.Array],
        dtype: number
        ) -> gym.spaces.Dict | gym.spaces.Box:
    if isinstance(spec, dict):
        return _dm_control_dict_spec_to_gym_dict(spec=spec, dtype=dtype)
    else:
        return _dm_control_array_spec_to_gym_box(spec=spec, dtype=dtype)


def extract_dict_observations_from_dm_control_timestep(
        timestep: TimeStep,
        observation_space: gym.spaces.Dict
        ) -> Dict[str, np.ndarray]:
    obs = timestep.observation
    dict_obs = dict()
    for key, value in obs.items():
        dict_obs[key] = value.flatten().astype(observation_space[key].dtype)
    return dict_obs


class DMC2GymWrapper(gym.Wrapper):
    metadata = {"render.modes": ['rgb_array']}

    def __init__(
            self,
            env: dm_control.composer.Environment,
            visual_observations: bool = False,
            frame_height: int = 400,
            frame_width: int = 400,
            camera_ids: List[int] | None = None,
            rendered_geom_groups: List[int] | None = None,
            rendered_site_groups: List[int] | None = None
            ) -> None:
        super().__init__(env)
        self.seed()

        self._visual_observations = visual_observations
        self._frame_height = frame_height
        self._frame_width = frame_width

        self.action_space: gym.spaces.Box = _dm_control_array_spec_to_gym_box(
                spec=self._env.action_spec(), dtype=np.float32
                )

        if visual_observations:
            self.observation_space = spaces.Box(
                    low=0, high=255, shape=[3, self._frame_height, self._frame_width], dtype=np.uint8
                    )
        else:
            self.observation_space = _dm_control_spec_to_gym_space(
                    spec=self._env.observation_spec(), dtype=np.float32
                    )

        self._camera_ids = camera_ids or [0]
        self._rendered_geom_groups = rendered_geom_groups or [1] * 6
        self._rendered_site_groups = rendered_site_groups or [1] * 6

    def __getattr__(
            self,
            name: str
            ) -> Any:
        return getattr(self._env, name)

    def _get_obs(
            self,
            time_step: TimeStep
            ) -> ObsType:
        if self._from_pixels:
            obs = self.render(height=self._frame_height, width=self._frame_width)
            obs = obs.transpose((2, 0, 1)).copy()
        else:
            obs = extract_dict_observations_from_dm_control_timestep(
                    timestep=time_step, observation_space=self.observation_space
                    )
        return obs

    def seed(
            self,
            seed: int = erpy_seed
            ) -> None:

        self.action_space.seed(seed)

        self.observation_space.seed(seed)

    def step(
            self,
            action: np.ndarray
            ) -> (ObsType, float, bool, bool, Dict[str, Any]):

        action = action.clip(min=self.action_space.low, max=self.action_space.high)

        time_step = self._env.step(action)
        reward = time_step.reward or 0
        terminated = time_step.last()
        obs = self._get_obs(time_step)

        info = {}
        try:
            info.update(self._env.task.get_info(time_step=time_step, physics=self._env.physics))
        except AttributeError:
            pass
        info['discount'] = time_step.discount

        return obs, reward, terminated, False, info

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None
            ) -> tuple[ObsType, dict[str, Any]]:
        time_step = self._env.reset()
        self.seed(seed)
        obs = self._get_obs(time_step)
        return obs, {}

    def render(
            self,
            mode: str = 'rgb_array',
            height: int | None = None,
            width: int | None = None,
            camera_ids: List[int] | None = None
            ) -> np.ndarray:

        camera_ids = camera_ids or self._camera_ids
        height = height or self._frame_height
        width = width or self._frame_width

        scene_option = MjvOption()
        scene_option.geomgroup = self.geomgroup
        scene_option.sitegroup = self.sitegroup

        frames = []
        for camera_id in camera_ids:
            frame = self._env.physics.render(
                    height=height, width=width, camera_id=camera_id, scene_option=scene_option
                    )
            frame = frame[:, :, ::-1]
            frames.append(frame)

        if len(frames) > 1:
            return np.hstack(frames)
        else:
            return frames[0]
