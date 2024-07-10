# Add necessary imports
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torchvision.transforms as transforms
from PIL import Image
import speech_recognition as sr
import pyaudio
import pywhatkit
import datetime
import pyttsx3
from pygame import mixer
import wikipedia
import time
import vlc
import random
import os
from pywttr import Wttr

# Function for Perceptron's voice
def say(text):
    engine.say(text)
    engine.runAndWait()

# Manual function for the weather
def weather():
    engine.say("The average temperature in Vera is " + forecast.weather[0].avgtemp_c + " degrees celsius")
    engine.runAndWait()

# Manual function for playing music
def sound(path):
    mixer.music.load(path)
    mixer.music.play()

# Function to generate responses
def generate_response(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
wttr = Wttr("Vera")
forecast = wttr.en()
engine = pyttsx3.init()
mixer.init()
recorder = sr.Recognizer()

# Load the trained model weights
model_save_path = "gpt2_dailydialog.pt"
model.load_state_dict(torch.load(model_save_path))
model.eval()

while True:
    try:
        with sr.Microphone() as source:                                                                       
            print("Speak:")                                                                                   
            audio = recorder.listen(source)
        print("Recognizing:")
        text = recorder.recognize_google(audio)

        # If perceptron is in the text continue
        if str("Perceptron") in text:
            sound("/home/adrian/Downloads/ding.mp3")
            text = text.replace("Perceptron", "")
            print("you said: " + text)

            # Plays music
            if "play music" in text:
                song = vlc.MediaPlayer("/home/adrian/Downloads/mercy.mp3")
                say("Ok, playing...")
                print("VLC is starting...")
                song.play()
                print("VLC is playing...")

            # Stops music
            elif "stop music" in text:
                say("Ok, stopping...")
                print("VLC is stopping...")
                song.stop()

            # Searches on google
            elif "Google" in text:
                say("Ok, searching...")
                pywhatkit.search(text)

            # Gets the time
            elif "time" in text:
                time = datetime.datetime.now().strftime("%H:%M")
                say("current time is " + time)

            # Searches for someone on wikipedia
            elif "who is" in text:
                person = text.replace("who is", "")
                info = wikipedia.summary(person, 1)
                say(info)

            # Says the weather
            elif "weather" in text:
                weather()

            # Even make a special feature to choose a random fart!!!
            elif "fart" in text:
                list1 = ["/home/adrian/Downloads/01.mp3",
                         "/home/adrian/Downloads/02.mp3",
                         "/home/adrian/Downloads/03.mp3",
                         "/home/adrian/Downloads/04.mp3",
                         "/home/adrian/Downloads/05.mp3"]
                ranfart = random.choice(list1)
                fart = vlc.MediaPlayer(ranfart)
                time.sleep(2)
                fart.play()

            elif "image" in text:
                say("Please input your image path")
                path = input("Input your image path: ")

    except sr.UnknownValueError:
        print("UnknownValueError")
        pass
    except sr.RequestError:
        print("RequestError")        
