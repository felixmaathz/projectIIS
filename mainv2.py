from time import sleep
from feat import Detector
from furhat_remote_api import FurhatRemoteAPI
import speech_recognition as sr
import threading
from threading import Lock
import pickle
import cv2

stop_thread = False
user_input = None
emotion = None
lock = Lock()

def load_furhat():
    FURHAT_IP = "127.0.1.1"
    furhat = FurhatRemoteAPI(FURHAT_IP)
    furhat.set_led(red=100, green=50, blue=50)
    FACES = {
        'Worker'    : 'Omar'
    }
    VOICES_EN = {
        'Worker'    : 'Brian'
    }
    furhat.set_face(character=FACES['Worker'], mask="Adult")
    furhat.set_voice(name=VOICES_EN['Worker'])
    return furhat


def bsay(line, fh):
    fh.say(text=line, blocking=True)

def recognize_speech():
    global user_input
    global stop_thread

    recognizer = sr.Recognizer()
    while True: 
        if stop_thread:
            break
        with sr.Microphone() as source:
            print("Talk!")
            audio = recognizer.listen(source, timeout=60)
        try:
            text = recognizer.recognize_google(audio)
            with lock:
                user_input = text
            print(f"You said: {text}")
        except sr.UnknownValueError:
            print("Could you repeat that?")
        except sr.RequestError as e:
            print(f"Request error: {e}")
    print("Thread finished")


def load_camera():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cam

def detect_emotion(detector, frame, model):
    global emotion
    faces = detector.detect_faces(frame)
    landmarks = detector.detect_landmarks(frame, faces)    
    action_units = detector.detect_aus(frame, landmarks)
    action_units = action_units[0]
    emotion = model.predict(action_units)
    return 





def main():

    global emotion
    global user_input
    global stop_thread

    model = pickle.load(open('model.sav', 'rb'))
    detector = Detector(device="cuda")


    cam = load_camera()
    print("Camera loaded...")

    furhat = load_furhat()
    print("Furhat loaded...")

    thread = threading.Thread(target=recognize_speech)
    thread.start()
    print("Thread started...")

    bsay("Hello, my name is Omar",furhat)

    detection_started = False

    try: 
        while True:

            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            if user_input and "hello" in user_input.lower():
                bsay("Hello",furhat)
                # Use a lock to prevent race conditions
                with lock:
                    user_input = None

            if user_input and "detect" in user_input.lower():
                detect_thread = threading.Thread(target=detect_emotion, args=(detector, frame, model))
                detect_thread.start()
                detection_started = True
                print("Detect thread started...")                
                with lock:
                    user_input = None
            
            if detection_started and not detect_thread.is_alive() and emotion:
                bsay(f"I think you are {emotion}",furhat)
                print("Detect thread: ", detect_thread.is_alive())
                emotion = None
                detection_started = False

            if user_input and "stop" in user_input.lower():
                bsay("Goodbye",furhat)
                stop_thread = True
                break


            if cv2.waitKey(1) == ord("q"):
                break

            cv2.imshow("frame", frame)

    finally:
        thread.join()
        print("Stopped thread")
        cam.release()
        cv2.destroyAllWindows()
        print("Closed camera")


if __name__ == "__main__":
    main()

