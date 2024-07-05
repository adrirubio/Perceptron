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

def say(text):
    engine.say(text)
    engine.runAndWait()

def weather():
    engine.say("The average temperature in Vera is " + forecast.weather[0].avgtemp_c + " degrees celsius")
    engine.runAndWait()

def sound(path):
    mixer.music.load(path)
    mixer.music.play()
