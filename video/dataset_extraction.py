import math
import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, frames_dir,start_sec,stop_sec):
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    stop_frame = int(stop_sec * fps)
    tot_frames = stop_frame - start_frame
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in tqdm(range(tot_frames)):
        success, image = vidcap.read()
        image = cv2.resize(image, (256, 256))
        cv2.imwrite(os.path.join(frames_dir, "{:d}.jpg".format(count)), image)
        count += 1
    return count

extract_frames('./video.mp4', './frames',start_sec=10,stop_sec=35*60)