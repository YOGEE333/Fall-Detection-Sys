import speech_recognition as sr
import pyttsx3
from telegram import Bot
import tracemalloc
import requests

tracemalloc.start()

recognizer = sr.Recognizer()

engine = pyttsx3.init()


telegram_bot_token = "7156811542:AAHuK_d-njPwXjBz8k9M25EWax0JnbX4l7A"
bot = Bot(token=telegram_bot_token)
chat_id = "1869333272" 

def listen():
    with sr.Microphone() as source:
        print("Listening for 'help'...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio).lower()
        print("You said:", query)
        return query
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
        return ""

def send_telegram_message(message):
    send_text = "https://api.telegram.org/bot" + telegram_bot_token + "/sendMessage?chat_id=" + chat_id + "&text=" + message
    responce = requests.get(send_text)
    return responce.json()
    

if __name__ == "__main__":
    while True:
        query = listen()
        if "help" in query:
            send_telegram_message("Someone needs help!(audio)")
            print("message sent")