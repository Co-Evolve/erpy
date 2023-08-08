from typing import Any, Callable, Iterable

import numpy as np
from dm_control import mjcf
from dm_control.composer.observation.observable import Generic, MJCFFeature
from dm_control.mjcf.physics import SynchronizingArrayWrapper
from dm_env import specs
from numpy.random import RandomState


class ConfinedMJCFFeature(MJCFFeature):
    def __init__(
            self,
            low: float,
            high: float,
            num_obs_per_element: int,
            kind: str,
            mjcf_element: mjcf.Element,
            update_interval: int = 1,
            buffer_size: int | None = None,
            delay: float | None = None,
            aggregator: str | Callable[[SynchronizingArrayWrapper, RandomState], np.ndarray] | None = None,
            corruptor: Callable[[SynchronizingArrayWrapper, RandomState], np.ndarray] | None = None,
            index: int | None = None
            ) -> None:
        super(ConfinedMJCFFeature, self).__init__(
                kind=kind,
                mjcf_element=mjcf_element,
                update_interval=update_interval,
                buffer_size=buffer_size,
                delay=delay,
                aggregator=aggregator,
                corruptor=corruptor,
                index=index
                )
        self._low = low
        self._high = high
        self._num_obs_per_element = num_obs_per_element

    @property
    def array_spec(
            self
            ) -> specs.BoundedArray:
        return specs.BoundedArray(
                shape=[self._num_obs_per_element * len(self._mjcf_element)],
                dtype=float,
                minimum=self._low,
                maximum=self._high
                )


class ConfinedObservable(Generic):
    def __init__(
            self,
            low: float,
            high: float,
            shape: Iterable,
            raw_observation_callable: Callable[[mjcf.Physics], Any]
            ):
        super(ConfinedObservable, self).__init__(raw_observation_callable)
        self._low = low
        self._high = high
        self._shape = shape

    @property
    def array_spec(
            self
            ) -> specs.BoundedArray:
        return specs.BoundedArray(
                shape=self._shape, dtype=float, minimum=self._low, maximum=self._high
                )
