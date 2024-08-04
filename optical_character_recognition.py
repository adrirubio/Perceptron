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
for inputs, targets in train_loader:
    print(inputs.shape)
    print(targets)
    break

# Display some examples from the test_loader
print("/nTesting batch examples")
for inputs, targets in test_loader:
    print(inputs.shape)
    print(targets)
    break
