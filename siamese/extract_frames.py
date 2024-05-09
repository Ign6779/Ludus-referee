import cv2
import os
import numpy as np
import random

# Specify the path to your video and the directory to save frames
video_path = r"C:\Users\ignac\Documents\InHolland\Year 3\Ludus project\dataset_ludus\2022-05-05 11.32.11_Camera 1_6.mov"
frames_save_path = r"C:\Users\ignac\Documents\InHolland\Year 3\Ludus project\temp"

if not os.path.exists(frames_save_path):
    os.makedirs(frames_save_path)

# Modes for frame extraction
modes = [30, 45, 30, 60, 15, 40, 30, 'random']

def extract_frame_pairs(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pairs_extracted = 0
    while pairs_extracted < 10:
        mode = random.choice(modes)
        if mode != 'random':
            # Ensure we have enough space for the pair
            max_start = max(0, total_frames - mode - 1)
            start_frame = random.randint(0, max_start)
            frame_indexes = [start_frame, start_frame + mode]
        else:
            frame_indexes = sorted(random.sample(range(total_frames), 2))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indexes[0])  # Go to the first frame of the pair
        
        for idx in frame_indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Go to the exact frame index
            ret, frame = cap.read()
            if not ret:
                break  # In case we can't read, skip this pair
            
            frame_save_path = os.path.join(save_dir, f"vid10_pair{pairs_extracted}_frame{idx}.jpg")
            cv2.imwrite(frame_save_path, frame)
        
        pairs_extracted += 1

    cap.release()

extract_frame_pairs(video_path, frames_save_path)
print("Extracted 10 pairs of frames with varying distances.")
