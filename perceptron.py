# Add necessary imports
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
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
