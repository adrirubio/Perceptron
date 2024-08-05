import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the IAM Handwriting dataset
dataset = load_dataset("iam_dataset")

# Define transformations for the images
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((128, 32)),
    transforms.ToTensor(),
])

# Function to transform images and prepare data for the train set
def transform_train_func(example):
    image = Image.open(example["image_path"]).convert("RGB")
    example["image"] = transform_train(image)
    return example

# Function to transform images and prepare data for the test set
def transform_test_func(example):
    image = Image.open(example["image_path"]).convert("RGB")
    example["image"] = transform_test(image)
    return example

train_dataset = dataset["train"].map(transform_train_func, remove_columns["image_path"], batched=False)
test_dataset = dataset["test"].map(transform_test_func, remove_columns["image_path"], batched=False)

# Check the first example in the training dataset
train_example = train_dataset[0]
print("Train example image shape: ", train_example["image"].shape)
print("Train example transcription: ", train_example["text"])

# Check the first example in the testing dataset
test_example = test_dataset[0]
print("Test example image shape: ", test_example["image"].shape)
print("Test example transcription: ", test_example["image"])

# Data loader
batch_size = 64
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True)

test_loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True)

# Display some examples from the train_loader
print("Training batch examples")
for batch in train_loader:
    inputs, targets = batch["image"], batch["label"]
    print(inputs.shape)
    print(targets)
    break

# Display some examples from the test_loader
print("\nTesting batch examples")
for batch in test_loader:
    inputs, targets = batch["image"], batch["label"]
    print(inputs.shape)
    print(targets)
    break

class CNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super(HandwritingRecognitionCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.pool = nn.MaxPool2d(2)
        
        # LSTM for sequence prediction
        self.lstm = nn.LSTM(input_size=512 * 4 * 4, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the feature maps
        
        # LSTM for sequence prediction
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the output of the last time step
        
        return x

# Example usage
num_classes = 100  # This should be the number of unique characters or classes in your dataset
model = CNN(num_classes)
