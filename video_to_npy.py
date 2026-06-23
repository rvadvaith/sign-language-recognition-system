import cv2
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector

DATASET_PATH = "videos"
SAVE_PATH = "dataset_npy"
FRAME_COUNT = 40  # Standardized to match your training/run scripts

# Initialize cvzone detector
detector = HandDetector(detectionCon=0.5, maxHands=2)

actions = sorted(os.listdir(DATASET_PATH))

for action in actions:
    action_path = os.path.join(DATASET_PATH, action)
    if not os.path.isdir(action_path): continue

    for video in os.listdir(action_path):
        if not video.endswith(".mp4"): continue

        cap = cv2.VideoCapture(os.path.join(action_path, video))
        sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Find hands with cvzone
            hands, img = detector.findHands(frame, draw=False)
            
            landmarks = []
            for i in range(2): # Logic for 2 hands (126 values total)
                if i < len(hands):
                    lmList = hands[i]["lmList"] # 21 points
                    wrist = lmList[0] # Root point for relative calculation
                    for lm in lmList:
                        # x, y, z relative to wrist
                        landmarks.extend([lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]])
                else:
                    landmarks.extend([0] * 63) # Padding for missing hand

            sequence.append(landmarks)
            if len(sequence) == FRAME_COUNT: break 

        cap.release()

        if len(sequence) == FRAME_COUNT:
            save_folder = os.path.join(SAVE_PATH, action)
            os.makedirs(save_folder, exist_ok=True)
            file_count = len(os.listdir(save_folder))
            np.save(os.path.join(save_folder, f"{file_count}.npy"), np.array(sequence))
            print(f"Processed: {action} - {video}")

print("Conversion complete.")