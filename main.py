import cv2
import opencv_jupyter_ui as jcv2
from feat import Detector
from IPython.display import Image
from feat.utils import FEAT_EMOTION_COLUMNS
from datasets import load_dataset

dataset = load_dataset("FER-Universe/DiffusionFER")

detector = Detector(device="cuda")


#User perception sub-system Gabriel

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
while True:
    ret, frame = cam.read()
    if not ret:
        print("OpenCV found an error reading the next frame.")
        break

    faces = detector.detect_faces(frame)
    landmarks = detector.detect_landmarks(frame, faces)
    emotions = detector.detect_emotions(frame, faces, landmarks)

    # The functions seem to assume a collection of images or frames. We acces "frame 0".
    faces = faces[0]
    landmarks = landmarks[0]
    emotions = emotions[0]

    strongest_emotion = emotions.argmax(axis=1)

    for (face, top_emo) in zip(faces, strongest_emotion):
        (x0, y0, x1, y1, p) = face
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 3)
        cv2.putText(frame, FEAT_EMOTION_COLUMNS[top_emo], (int(x0), int(y0 - 10)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

    jcv2.imshow("Emotion Detection", frame)

    key = jcv2.waitKey(1) & 0xFF
    if key == 27: # ESC pressed
        break

cam.release()
jcv2.destroyAllWindows()