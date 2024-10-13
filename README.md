# Perceptron 🧠

### The Ultimate AI-Powered Assistant
***The Goal***

**Perceptron** aims to be a cutting-edge AI-powered assistant that will seamlessly combine multiple AI models into a single, powerful tool. As I continue to develop it, the goal is that by simply invoking the name "Perceptron," you'll be able to trigger a wide range of actions—from engaging in conversations to performing complex image analysis. I am working towards integrating models for transfer learning, sentiment analysis, image recognition, object detection, and OCR (Optical Character Recognition) to ensure that your interactions with Perceptron are as seamless and intelligent as possible.

***Some features are still a work in progress***
***Currently training models and preparing the inference code (perceptron.py)***

---

## 🚀 Features

- **Conversational AI 🤖**: Chat with Perceptron using natural language. Trigger actions like playing music, searching on Google/Wikipedia, getting the time, or checking the weather.
  
- **Image Recognition 💡**: Say "image" and provide a file path to get predictions on what the image contains.

- **Sentiment Analysis --> (work in progress)✨**: Perceptron analyzes the sentiment of the conversation, predicting emotions like happiness, sadness, and more.

- **Object Detection 📸**: Detect and highlight objects in images with bounding boxes.

- **Optical Character Recognition (OCR) --> (work in progress) 🔍**: Extract and identify text from images.

---

## ✨ Manual Commands and Actions

Perceptron is not only capable of holding conversations, but it can also perform a variety of specific tasks when triggered by certain commands. Below are some of the manual commands you can use to interact with Perceptron:

### 1. **Check the Time 🕒**
   - **Command**: `"time"`
   - **Description**: Perceptron will respond with the current time.

### 2. **Check the Weather 🌡️**
   - **Command**: `"weather"`
   - **Description**: Get the current weather conditions in your location or in a specified city.
   
### 3. **Search on Wikipedia 📖**
   - **Command**: `"wiki [topic]"`
   - **Description**: Perceptron will fetch a summary of the topic from Wikipedia.

### 4. **Search on Google 🌐**
   - **Command**: `"Search Google for [query]"`
   - **Description**: Initiates a Google search and opens the search.

### 5. **Play Music 🎧**
   - **Command**: `"Play music"`
   - **Description**: Perceptron will start playing music from a specified local library.

### 6. **Stop Music 🎧**
   - **Command**: `"Stop music"`
   - **Description**: Stops the currently playing music.

---

## 🛠️ How to Run the Project

### Installation

1. **Clone the Repository**:
    - To get started, clone this repository to your local machine using the following command:

      ```bash
      git clone https://github.com/adrirubio/Perceptron.git

2. **Run Perceptron**:
    - Next run the following commands to start using Perceptron:

      ```python
      pip install transformers
      pip install Pillow
      pip install speechrecognition
      sudo apt-get install portaudio19-dev
      pip install pyaudio
      pip install pywhatkit
      pip install pyttsx3
      pip install pygame
      pip install wikipedia-api
      pip install python-vlc
      pip install pywttr
      python Perceptron.py
      ```

3. **Enjoy!**:
    - Your Perceptron AI assistant is now up and running!!!
    - Please note that the AI models that work are the following (the others are still being built):
      `"CNN_model"`
      `"transfer_learning_model"`
      `"object_detection_model"`
    - Though please do be aware the three models named above train and work flawlessly, but the actual inference code (Perceptron.py) is a bit outdated and as lately I have       been focusing on the AI models the inference code and manual commands might have some bugs.
    - Sorry that Perceptron is not yet fully functional.

---

## 🎥 Demo

Here’s a demo of Perceptron in action:

### 1. CNN Demo 📸
[Epic CNN Demo!](https://cloud-6knwch9wm-hack-club-bot.vercel.app/0captura_de_pantalla_2024-08-27_162100.png)

### 2. Object Detection Demo 📸

--> ***Transfer learning model and manual commands demo coming soon...***

---

## 📖 Background


**Perceptron** is on its way to becoming an interactive and intelligent assistant by harnessing the power of various AI models. Whether it's analyzing images, detecting objects, recognizing text, or having a conversation, I'm working to ensure Perceptron will seamlessly integrate these functionalities. This ongoing project highlights the potential of AI to enhance user experiences through natural language processing and advanced image analysis.

---

## 📄 License

This project is licensed under the MIT License

---

## 🤝 Contact
For questions or support, please open an issue or contact adrian.rubio.punal@gmail.com
