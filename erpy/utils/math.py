from __future__ import division

from functools import reduce
from typing import List

import numpy as np


def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]


def generate_numbers_target_mean(wanted_avg: float, numbers_to_generate: int, start: int, end: int,
                                 random_state: np.random.RandomState) -> List[int]:
    rng = [i for i in range(start, end)]
    initial_selection = [random_state.choice(rng) for _ in range(numbers_to_generate)]
    initial_avg = reduce(lambda x, y: x + y, initial_selection) / float(numbers_to_generate)
    if initial_avg == wanted_avg:
        return initial_selection

    off = abs(initial_avg - wanted_avg)
    manipulation = off * numbers_to_generate

    sign = -1 if initial_avg > wanted_avg else 1

    manipulation_action = dict()
    acceptable_indices = list(range(numbers_to_generate))
    while manipulation > 0:
        random_index = random_state.choice(acceptable_indices)
        factor = manipulation_action[random_index] if random_index in manipulation_action else 0
        after_manipulation = initial_selection[random_index] + factor + sign * 1
        if start <= after_manipulation < end:
            if random_index in manipulation_action:
                manipulation_action[random_index] += sign * 1
                manipulation -= 1
            else:
                manipulation_action[random_index] = sign * 1
                manipulation -= 1
        else:
            acceptable_indices.remove(random_index)

    for key in manipulation_action:
        initial_selection[key] += manipulation_action[key]

    return initial_selection

