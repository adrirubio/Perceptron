# Add necessary imports
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertForSequenceClassification
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

# Function to generate responses using Transfer Learning model
def generate_response(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to generate predict images using CNN model
def predict_images(image_path, model, transformer):
    # Load and transform image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    # Set model to evaluation mode
    model.eval()

    # Perform inference
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    # Return predicted class
    predicted_class = predicted.item()
    return predicted_class

# Function to predict the sentiment of a single sentence
def predict_sentence(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)
    return prediction.item()

def infer_and_display(image_path, model, transform):
    # Load the image
    image = Image.open(image_path).comavert("RGB")

    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        _, bbox_preds = model(image_tensor)
        bbox_preds = bbox_preds.cpu().squeeze().tolist()

    # Ensure bbox_preds have correct format
    if len(bbox_preds) != 4:
        print("Error: Bounding box predictions are not in correct format")
        return

    # Convert bbox from [xmin, ymin, width, height] to [xmin, ymin, xmax, ymax]
    bbox = [bbox_preds[0], bbox_preds[1], bbox_preds[0] + bbox_preds[2], bbox_preds[1] + bbox_preds[3]]

    # Draw bounding box on image
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red", width=3)

    # Display image
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Define the CNN model class (same as in the training script)
class Object_Detection(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(1024, num_classes)

        self.fc3 = nn.Linear(128 * 4 * 4, 1024)
        self.fc4 = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_flat = x.view(x.size(0), -1)

        x_class = self.dropout(x_flat)
        x_class = self.relu(self.fc1(x_class))
        x_class = self.dropout(x_class)
        class_logits = self.fc2(x_class)

        x_bbox = self.dropout(x_flat)
        x_bbox = self.relu(self.fc3(x_bbox))
        x_bbox = self.dropout(x_bbox)
        bbox_coordinates = self.fc4(x_bbox)

        return class_logits, bbox_coordinates

class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )

        # Define the dense (fully connected) layers
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(1024, K)

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the output for the dense layers
        x = x.view(x.size(0), -1)

        # Forward pass through the dense layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

wttr = Wttr("Vera")
forecast = wttr.en()
engine = pyttsx3.init()
mixer.init()
recorder = sr.Recognizer()
# Perform transformations on input image
transformer_test = transforms.ToTensor()

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

            # Predicts the class of an image
            elif "image" in text:
                # Load the CNN trained model
                model_save_path = "cnn_cifar100_model.pth"

                K = 100 # Num classes
                model = CNN(K)  # Instantiate your CNN model
                model.load_state_dict(torch.load(model_save_path))
                model.eval()
                
                say("Please input your image path")
                image_path = input("Input your image path: ")
                predicted_class = predict_image(image_path, model, transformer_test)
                print(f"The predicted class for the image is: {predicted_class}")
                say(f"The predicted class for the image is: {predicted_class}")

                # Load the Object Detection model
                model_path = "object_detection_model.pth"

                num_classes = 91 # 80 classes + 1 background for COCO
                model = Object_Detection(num_classes)
                model.load_state_dict(torch.load(model_path))
                model.eval()

                infer_and_display(image_path, model, transformer_test, device)

            else:
                # Load GPT-2 tokenizer and model
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                model = GPT2LMHeadModel.from_pretrained('gpt2')
                
                # Load the trained model weights
                model_save_path = "gpt2_dailydialog.pt"
                model.load_state_dict(torch.load(model_save_path))
                model.eval()

                # Uses the trained model if it shouldn't be a manual response
                response = generate_response(text, model, tokenizer)
                print("Response:", response)
                say(response)

                # Paths to the saved model and tokenizer
                model_save_path = "sentiment_analysis_model.pt"
                tokenizer_save_path = "sentiment_analysis_tokenizer"

                # Load the tokenizer
                tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)

                # Load the sentiment analisis mode"
                model = BertForSequenceClassification.from_pretrained("bert-base-uncased·, num_labels=2)
                model.load_state_dict(torch.load(model_save_path))
                model.eval()

                # Also tells you how you are feeling
                prediction = predict_sentence(model, tokenizer, text)
                print(f"Sentence: {text} => Prediction: {"Positive" if prediction == 1 else "Negative"}")
                say(f"Sentence: {text} => Prediction: {"Positive" if prediction == 1 else "Negative"}")
                
    except sr.UnknownValueError:
        print("UnknownValueError")
        pass
    except sr.RequestError:
        print("RequestError")        
