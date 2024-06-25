# Perceptron: A Transfer Learning AI

## Overview
Perceptron is a Transfer Learning AI designed to assist with various tasks, triggered by the keyword "Perceptron". It utilizes state-of-the-art transfer learning techniques to adapt pre-trained models to new datasets and tasks efficiently.

## Features
- **Trigger-based Activation**: Perceptron listens for the keyword "Perceptron" to activate its functions.
- **Transfer Learning**: Leverages pre-trained models to quickly adapt to new tasks.
- **Customizable Dataset**: Uses a tailored dataset to optimize performance for specific applications.

## Dataset
Perceptron is trained on a unique dataset named `PerceptronDataset`. This dataset includes a variety of data to ensure the model is robust and versatile. Here are the details:

- **Sources**: The dataset is collected from multiple sources including text, audio, and image data.
- **Categories**:
  - **Text**: Includes articles, blog posts, and technical documents.
  - **Audio**: Contains recordings of various speech patterns and accents.
  - **Images**: Comprises labeled images for object recognition tasks.
- **Size**: Approximately 10,000 samples per category.
- **Format**: Stored in CSV files for text and audio metadata, and PNG/JPEG for images.

## Installation
To get started with Perceptron, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/perceptron.git
    cd perceptron
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset**:
    ```bash
    bash download_dataset.sh
    ```

## Usage
Perceptron is designed to run continuously and be triggered by the keyword "Perceptron". Here’s how you can set it up:

1. **Run the main script**:
    ```bash
    python main.py
    ```

2. **Cron Job Setup**:
    To ensure Perceptron is always running, set up a cron job:
    ```bash
    crontab -e
    ```
    Add the following line to run the script every minute:
    ```bash
    * * * * * /usr/bin/python /path_to_your_script/main.py
    ```

## Contributing
We welcome contributions to improve Perceptron! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License

## Contact
For questions or support, please open an issue or contact adrian.rubio.punal@Gmail.com
