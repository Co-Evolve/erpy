from dm_control.composer.observation.observable import MJCFFeature, Generic
from dm_env import specs


class ConfinedMJCFFeature(MJCFFeature):
    def __init__(self, low, high, num_obs_per_element, kind, mjcf_element, update_interval=1,
                 buffer_size=None, delay=None,
                 aggregator=None, corruptor=None, index=None) -> None:
        super(ConfinedMJCFFeature, self).__init__(kind=kind,
                                                  mjcf_element=mjcf_element,
                                                  update_interval=update_interval,
                                                  buffer_size=buffer_size,
                                                  delay=delay,
                                                  aggregator=aggregator,
                                                  corruptor=corruptor,
                                                  index=index)
        self._low = low
        self._high = high
        self._num_obs_per_element = num_obs_per_element

    @property
    def array_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=[self._num_obs_per_element * len(self._mjcf_element)],
                                  dtype=float,
                                  minimum=self._low,
                                  maximum=self._high
                                  )


class ConfinedObservable(Generic):
    def __init__(self, low, high, shape, raw_observation_callable):
        super(ConfinedObservable, self).__init__(raw_observation_callable)
        self._low = low
        self._high = high
        self._shape = shape

    @property
    def array_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=self._shape,
                                  dtype=float,
                                  minimum=self._low,
                                  maximum=self._high)
