import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from erpy.utils.video import create_video

if __name__ == '__main__':
    base_path = Path("/Users/driesmarzougui/phd/experiments/BRB/data/ventral_curvature_archive")
    runs = ['celestial-blaze-211', 'resilient-dawn-252', 'solar-frog-261', 'sandy-bush-269']

    frames = []
    for run in runs:
        path = base_path / run / 'archive'
        images = [img for img in os.listdir(path) if img.endswith("png")]
        sorted_image_indices = np.argsort([int(img.split('_')[3]) for img in images])
        images = [images[index] for index in sorted_image_indices]

        for image in tqdm(images, desc=f'Reading images from {run}'):
            frame = cv2.imread(str(path / image))
            frames.append(frame)


    cleaned_frames = []
    previous_frame = None
    for frame in frames:
        if previous_frame is None or not np.all(frame == previous_frame):
            cleaned_frames.append(frame)
        previous_frame = frame


    target_time = 15
    num_frames = len(cleaned_frames)
    framerate = num_frames / target_time
    create_video(cleaned_frames, framerate, 'archive_filling_15s_latest_cleaned.mp4')


    target_time = 5
    num_frames = len(cleaned_frames)
    framerate = num_frames / target_time
    create_video(cleaned_frames, framerate, 'archive_filling_5s_latest_cleaned.mp4')

    target_time = 10
    num_frames = len(frames)
    framerate = num_frames / target_time
    create_video(frames, framerate, 'archive_filling_10s_latest.mp4')


    target_time = 5
    num_frames = len(frames)
    framerate = num_frames / target_time
    create_video(frames, framerate, 'archive_filling_5s_latest.mp4')





