import numpy as np

seed = 42
random_state = np.random.RandomState(seed=42)


def set_random_state(seed_value: int):
    global seed, random_state
    seed = seed_value
    random_state.seed(seed_value)
