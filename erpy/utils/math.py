from __future__ import division

import numpy as np


def renormalize(data: np.ndarray, original_range: np.ndarray, target_range: np.ndarray) -> np.ndarray:
    delta1 = original_range[1] - original_range[0]
    delta2 = target_range[1] - target_range[0]

    return (delta2 * (data - original_range[0]) / delta1) + target_range[0]
