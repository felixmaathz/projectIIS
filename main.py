from time import sleep
from furhat_remote_api import FurhatRemoteAPI
import speech_recognition as sr
import cv2
import opencv_jupyter_ui as jcv2
from feat import Detector
from feat.utils import FEAT_EMOTION_COLUMNS
print("hej gabbe")

# Detector choice
detector = Detector(device="cuda")

# Furhat IP address
FURHAT_IP = "127.0.1.1"

# Connect to Furhat
furhat = FurhatRemoteAPI(FURHAT_IP)
furhat.set_led(red=100, green=50, blue=50)

# Furhat faces and voices
FACES = {'TheJoker': 'Brooklyn'}
VOICES_EN = {'TheJoker': 'GregoryNeural'}

# Furhat speech
def bsay(line):
    furhat.say(text=line, blocking=True)
    sleep(1)  # Add a delay of 1 second after each Furhat command

# Speech recognition setup
recognizer = sr.Recognizer()
with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source)

# Function to recognize speech
def recognize_speech():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, timeout=5)
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio.")
        return None

# Function for emotion detection
def detect_emotion(frame):
    faces = detector.detect_faces(frame)
    landmarks = detector.detect_landmarks(frame, faces)
    emotions = detector.detect_emotions(frame, faces, landmarks)

    # The functions assume a collection of images or frames. Access "frame 0".
    faces = faces[0]
    emotions = emotions[0]

    strongest_emotion = emotions.argmax(axis=1)

    for (face, top_emo) in zip(faces, strongest_emotion):
        (x0, y0, x1, y1, p) = face
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 3)
        cv2.putText(frame, FEAT_EMOTION_COLUMNS[top_emo], (int(x0), int(y0 - 10)),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

        # Return the detected emotion
        return FEAT_EMOTION_COLUMNS[top_emo]


# Function to react to speech
def react_to_speech(text):
    if text:
        print(f"You: {text}")
        if "you wanted me here i am" in text.lower():
            bsay("I wanted to see what you'd do...and you didn't disappoint. "
                 "You let 5 people die. Then you let Dent take your place. Even to a guy like me, that's cold.")
            furhat.gesture(name='Surprise')

            

# Function to react to emotion
def react_to_emotion(detected_emotion):
            
            print(f"Detected emotion: {detected_emotion}")
    # Adjust Furhat's behavior based on the detected emotion
            if detected_emotion == "happiness":
                bsay("HAPPY")
                furhat.gesture(name='HappyGesture')
            elif detected_emotion == "sadness":
                bsay("SAD")
                furhat.gesture(name='SadGesture')
            elif detected_emotion == "neutral":
                bsay("You are neutral")
                furhat.gesture(name='Wink')
            # Add more conditions based on different emotions

# Main interaction function
def interaction():
    furhat.set_face(character=FACES['TheJoker'], mask="Adult")
    furhat.set_voice(name=VOICES_EN['TheJoker'])
    bsay("Hi")
    furhat.gesture(name='ExpressDisgust')
    while True:
        ret, frame = cam.read()
        if not ret:
            print("OpenCV found an error reading the next frame.")
            break

        detected_emotion = detect_emotion(frame)
        react_to_speech(recognize_speech())
        react_to_emotion(detected_emotion)

        jcv2.imshow("Emotion Detection", frame)

        key = jcv2.waitKey(1) & 0xFF
        if key == 27:  # ESC pressed
            break

# Set up camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Run the main interaction loop
if __name__ == '__main__':
    interaction()

# Release resources
cam.release()
jcv2.destroyAllWindows()