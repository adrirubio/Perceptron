# Perceptron 🧠

### The Ultimate AI-Powered Assistant
***The Goal***

**Perceptron** is a cutting-edge AI-powered assistant that combines multiple AI models into a single, powerful tool. By simply invoking its name, "Perceptron," you can trigger various actions—from having a conversation to performing complex image analysis. With integrated models for sentiment analysis, image recognition, object detection, and OCR (Optical Character Recognition), Perceptron is designed to make your interactions as seamless and intelligent as possible.

***Some features are still work in progress***

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
    - Open [Google Colab](https://colab.research.google.com/) in your browser.
    - Start a new notebook and run the following code to clone the repository and navigate to the `"Perceptron"` directory:

      ```python
      !git clone https://github.com/adrirubio/Perceptron.git
      %cd perceptron
      ```

2. **Load the trained models**:
   - Run the following code to load the weights:

   


4. **Run Perceptron**:
    - Next run the following script to start using Perceptron:

      ```python
      !pip install transformers
      !pip install Pillow
      !pip install speechrecognition
      !sudo apt-get install portaudio19-dev
      !pip install pyaudio
      !pip install pywhatkit
      !pip install pyttsx3
      !pip install pygame
      !pip install wikipedia-api
      !pip install python-vlc
      !pip install pywttr
      %run Perceptron.py
      ```

5. **Enjoy!**:
    - Your Perceptron AI assistant is now up and running!!!
    - Though please do be aware the three models named above train and work flawlessly, but the actual inference code (Perceptron.py) is a bit outdated and as lately I have been working on the AI models the inference code and manual commands might have some flaws.

---

## 🎥 Demo

Here’s a demo of Perceptron in action:

### 1. CNN Demo 📸

### 2. Object Detection Demo 📸

--> ***Transfer learning model and manual commands demo coming soon...***

---

## 📖 Background

Perceptron brings together the power of various AI models to create an interactive, and intelligent assistant. Whether you're analyzing images, detecting objects, recognizing text, or simply having a conversation, Perceptron is designed to provide seamless integration of these functionalities. This project demonstrates the potential of AI to enhance user experiences through natural language processing and advanced image analysis.

---

## 📄 License

This project is licensed under the MIT License

---

## 🤝 Contact
For questions or support, please open an issue or contact adrian.rubio.punal@gmail.com
