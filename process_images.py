import pandas as pd
import os
import cv2
from feat import Detector

subdirectories = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

file_path = os.getcwd() + "/DiffusionEmotion_S/original/"

df = pd.DataFrame()
detector = Detector(device="cuda")

for subdirectory in subdirectories:
    subdirectory_path = file_path + subdirectory + "/" + subdirectory
    for file in os.listdir(subdirectory_path):
        image_path = subdirectory_path + "/" + file
        detection = detector.detect_image(image_path).aus
        temp_df = pd.DataFrame(columns=["label"])
        temp_df["label"] = [subdirectory]
        temp_df = pd.concat([temp_df, pd.DataFrame(detection, columns=detection.keys())], axis=1)
        df = pd.concat([df, temp_df], ignore_index=True)
        print(df)

df.to_csv("data.csv")
        