from typing import List

import cv2
import numpy as np


def create_video(
        frames: List[np.ndarray],
        framerate: float,
        out_path: str
        ) -> None:
    height, width, _ = frames[0].shape
    size = (width, height)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), framerate, size)
    for frame in frames:
        writer.write(frame)
    writer.release()
