from time import sleep
from feat import Detector
from furhat_remote_api import FurhatRemoteAPI
import speech_recognition as sr
import threading
from threading import Lock
import pickle
import cv2


# Global variables, used between threads
stop_thread = False
user_input = None
emotion = None

# Lock to prevent race conditions
lock = Lock()

# From lab 3
def load_furhat():
    FURHAT_IP = "127.0.1.1"
    furhat = FurhatRemoteAPI(FURHAT_IP)
    furhat.set_led(red=100, green=50, blue=50)
    FACES = {
        'bartender'    : 'Omar'
    }
    VOICES_EN = {
        'bartender'    : 'Matthew'
    }
    furhat.set_face(character=FACES['bartender'], mask="Adult")
    furhat.set_voice(name=VOICES_EN['bartender'])
    return furhat

def bsay(line, fh):
    fh.say(text=line, blocking=False)

def recognize_speech():

    # Global variables
    global user_input
    global stop_thread

    # Speech recognition, from lab 3
    recognizer = sr.Recognizer()
    while True: 

        if stop_thread:
            break

        with sr.Microphone() as source:
            sleep(2)
            print("Talk!")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
        try:
            print("Recognizing...")
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
    # From lab 1
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cam

def detect_emotion(detector, frame, model):
    # Global variable
    global emotion

    # Detect emotion, from lab 1
    faces = detector.detect_faces(frame)
    landmarks = detector.detect_landmarks(frame, faces)    
    action_units = detector.detect_aus(frame, landmarks)
    action_units = action_units[0]
    emotion = model.predict(action_units)
    with lock:
        emotion = emotion[0]
    print(f"Detected emotion: {emotion}")
    return # Closes thread

def conversation(text,furhat,responses,emotion):
    # Rule based system to respond to the user
    if text:
        if emotion == "happy":
            if "hello" in text.lower():
                bsay(responses["happy"][0],furhat)
            elif "thank you" in text.lower():
                bsay(responses["happy"][1],furhat)
            elif "my son just got engaged" in text.lower():
                bsay(responses["happy"][2],furhat)
            elif "do you have any drink recommendations" in text.lower():
                bsay(responses["happy"][3],furhat)

        elif emotion == "neutral":
            if "hello" in text.lower():
                bsay(responses["neutral"][0],furhat)
            elif "thank you" in text.lower():
                bsay(responses["neutral"][1],furhat)
            elif "correct i just visited the sauna" in text.lower():
                bsay(responses["neutral"][2],furhat)
            elif "do you have any drink recommendations" in text.lower():
                bsay(responses["neutral"][3],furhat)
                
        elif emotion == "angry":
            if "hello" in text.lower():
                bsay(responses["angry"][0],furhat)
            elif "whatever" in text.lower():
                bsay(responses["angry"][1],furhat)
            elif "my boss just gave me a lot of work during the holiday season" in text.lower():
                bsay(responses["angry"][2],furhat)
            elif "give me something to drink now" in text.lower():
                bsay(responses["angry"][3],furhat)



def main():

    # Global variables
    # Used between threads
    global emotion
    global user_input
    global stop_thread

    model = pickle.load(open('model.sav', 'rb'))
    detector = Detector(device="cuda")

    # Load furhat and camera
    furhat = load_furhat()
    print("Furhat loaded...")

    cam = load_camera()
    print("Camera loaded...")

    # Start the speech recognition thread
    thread = threading.Thread(target=recognize_speech, daemon=True)
    thread.start()
    print("Thread started...")

    bsay("Hello, my name is Omar",furhat)

    detection_started = False


    # Responses to different emotions
    emotion_responses = {
        "happy" : ["Welcome to DrinkTown, have a seat!",
                   "You seem pretty happy today, how come?",
                   "That’s great news, I hope he will live a happy life with his fiancee!",
                   "Here’s a glass of champagne, it’s on the house!"],
        "angry" : ["Welcome to DrinkTown, have a seat!",
                   "You seem a little angry, whats going on?",
                   "I’m sorry to hear that, I hope you will find time to visit your family",
                   "I’m sorry but I don't. I can't serve you anything since you seem a bit agitated, but here is a glass of water to calm your nerves."
                   ],
        "neutral" : ["Welcome to DrinkTown, have a seat!",
                     "You seem pretty calm today, why’s that?",
                     "That’s great to hear, you have to prioritize your health over anything!",
                     "Here’s a local lager to cool off, enjoy!"
                     ],
    }

    try: 

        # Main loop
        while True:
            
            # Read frame from camera, from lab 1
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            # If the user says something, detect the emotion and store in global variable
            if user_input and not detection_started:
                if "stop" in user_input.lower():
                    break
                detect_thread = threading.Thread(target=detect_emotion, args=(detector, frame, model))
                detect_thread.start()
                detection_started = True
                print("Detect thread started...")         


            # Check if the detection thread has finished, ie. emotion has been detected
            if emotion and user_input:

                # If the user says "reset", reset the emotion
                # Used for testing
                if "reset" in user_input.lower():
                    emotion = None
                    detection_started = False
                    with lock:
                        user_input = None
                    continue

                # If the user says "stop", stop the program
                if "stop" in user_input.lower():
                    stop_thread = True
                    break

                # Respond to the user based on the emotion and the user input
                conversation(user_input,furhat,emotion_responses,emotion)

                with lock:
                    user_input = None
            
            # If the user says "stop", stop the program
            if user_input and "stop" in user_input.lower():
                bsay("Goodbye",furhat)
                stop_thread = True
                break

            if cv2.waitKey(1) == ord("q"):
                print("Pressed q")
                break


            # Show the frame
            cv2.imshow("frame", frame)

    finally:
        # Stop the threads
        thread.join()
        print("Stopped thread")
        cam.release()
        cv2.destroyAllWindows()
        print("Closed camera")


if __name__ == "__main__":
    main()

