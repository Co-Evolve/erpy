from __future__ import division

import numpy as np


def rescale(
        data: np.ndarray,
        original_range: np.ndarray | (float, float),
        target_range: np.ndarray | (float, float)
        ) -> np.ndarray:
    delta1 = original_range[1] - original_range[0]
    delta2 = target_range[1] - target_range[0]

    return (delta2 * (data - original_range[0]) / delta1) + target_range[0]
