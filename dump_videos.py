import cv2
from tqdm.auto import tqdm
import numpy as np
import glob
import os
import gc

files = glob.glob("ydata-tvsum50-v1_1/video/*.mp4")
for file in tqdm(files):
    prefix = file.split("/")[-1].split(".")[0]
    save_path = f"videos_npy/{prefix}.npy"
    if os.path.exists(save_path):
        continue
    frames = []
    cap = cv2.VideoCapture(file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    frames = np.stack(frames)
    np.save(save_path, frames)
    del frames
    gc.collect()
