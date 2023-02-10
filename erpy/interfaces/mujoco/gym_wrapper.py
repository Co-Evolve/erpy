"""
Source: https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
Code was slightly adapted.
"""
from typing import Dict

import gym
import numpy as np
from dm_control.mujoco import wrapper
from dm_env import specs, TimeStep
from gym import core, spaces


def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)

    try:
        low = np.concatenate(mins, axis=0).astype(dtype)
        high = np.concatenate(maxs, axis=0).astype(dtype)
    except ValueError:
        low = np.array([])
        high = np.array([])
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def _spec_to_dict(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    d = dict()
    for key, array in spec.items():
        mn, mx = extract_min_max(array)
        d[key] = spaces.Box(low=mn, high=mx, dtype=dtype)

    return spaces.Dict(d)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    try:
        return np.concatenate(obs_pieces, axis=0)
    except ValueError:
        return np.array([])


def get_clean_obs(timestep: TimeStep, dtype = np.float32) -> Dict[str, np.ndarray]:
    obs = timestep.observation
    cleaned_obs = dict()
    for key, value in obs.items():
        cleaned_obs[key] = value.flatten().astype(dtype)
    return cleaned_obs


def vectorize_observations(observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    vectorized_obs = dict()
    for key, value in observations.items():
        vectorized_obs[key] = value[None]
    return vectorized_obs


class DMC2GymWrapper(core.Env):
    metadata = {"render.modes": ['rgb_array']}

    def __init__(
            self,
            env,
            seed,
            from_pixels=False,
            height=400,
            width=400,
            camera_ids=None,
            channels_first=True,
            geomgroup=None,
            sitegroup=None
    ):
        self._from_pixels = from_pixels
        if camera_ids is None:
            self.camera_ids = [0]
        else:
            self._camera_ids = camera_ids

        self._height = height
        self._width = width
        self._channels_first = channels_first

        # create task
        self._env = env

        self.action_space = _spec_to_box([self._env.action_spec()], np.float32)

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self.observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self.observation_space = _spec_to_dict(
                self._env.observation_spec(),
                np.float32
            )

        # set seed
        self.seed(seed=seed)

        self.geomgroup = geomgroup
        if self.geomgroup is None:
            self.geomgroup = [1] * 6
        self.sitegroup = sitegroup
        if self.sitegroup is None:
            self.sitegroup = [1] * 6

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
            )
            if self._channels_first:
                obs = obs.transpose((2, 0, 1)).copy()
        else:
            obs = get_clean_obs(time_step)
        return obs

    @property
    def reward_range(self):
        return self._env.reward_spec()

    def seed(self, seed=None):
        if seed is None:
            seed = 42
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def step(self, action: np.ndarray):
        action = action.clip(min=self.action_space.low, max=self.action_space.high)
        reward = 0
        time_step = self._env.step(action)
        info = {}
        reward += time_step.reward or 0
        done = time_step.last()
        obs = self._get_obs(time_step)

        try:
            info.update(self._env.task.get_info(time_step=time_step, physics=self._env.physics))
        except AttributeError:
            pass
        info['discount'] = time_step.discount

        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_ids=None):
        if camera_ids is None:
            camera_ids = self._camera_ids

        height = height or self._height
        width = width or self._width
        pixels_default = []
        pixels_specific = []
        for camera_id in camera_ids:
            default_frame = self._env.physics.render(
                height=height, width=width, camera_id=camera_id,
            )
            default_frame = default_frame[:, :, ::-1]
            pixels_default.append(default_frame)

            if self.geomgroup != [1] * 6 or self.sitegroup != [1] * 6:
                scene_option = wrapper.core.MjvOption()
                scene_option.geomgroup = self.geomgroup
                scene_option.sitegroup = self.sitegroup

                frame = self._env.physics.render(
                    height=height, width=width, camera_id=camera_id,
                    scene_option=scene_option
                )
                frame = frame[:, :, ::-1]
                pixels_specific.append(frame)

        view = np.hstack(pixels_default)
        if pixels_specific:
            specific_view = np.hstack(pixels_specific)
            view = np.vstack((view, specific_view))
        return view


class HistoryWrapper(gym.Wrapper):
    """
    Stack past observations and actions to give an history to the agent.
    :param env:
    :param horizon:Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 2):
        assert isinstance(env.observation_space, gym.spaces.Box)

        wrapped_obs_space = env.observation_space
        wrapped_action_space = env.action_space

        # TODO: double check, it seems wrong when we have different low and highs
        low_obs = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        high_obs = np.repeat(wrapped_obs_space.high, horizon, axis=-1)

        low_action = np.repeat(wrapped_action_space.low, horizon, axis=-1)
        high_action = np.repeat(wrapped_action_space.high, horizon, axis=-1)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)

        super().__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self):
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self):
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs = self.env.reset()
        self.obs_history[..., -obs.shape[-1]:] = obs
        return self._create_obs_from_history()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1]:] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1]:] = action
        return self._create_obs_from_history(), reward, done, info
