from furhat_remote_api import FurhatRemoteAPI
import speech_recognition as sr
import threading

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

furhat = load_furhat()

def recognize_speech():


    recognizer = sr.Recognizer()
    while True: 


        with sr.Microphone() as source:
            print("Talk!")
            audio = recognizer.listen(source, timeout=10)
        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            
            print(f"You said: {text}")
        except sr.UnknownValueError:
            print("Could you repeat that?")
        except sr.RequestError as e:
            print(f"Request error: {e}")

thread = threading.Thread(target=recognize_speech)
thread.start()




