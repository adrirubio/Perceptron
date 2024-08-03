import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
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
def transform_train(example):
    image = Image.open(example["image_path"]).convert("RGB")
    example["image"] = transform(image)
    return example

# Function to transform images and prepare data for the test set
def transform_test(example):
    image = Image.open(example["image_path"]).convert("RGB")
    example["image"] = transform(image)
    return example

train_dataset = dataset["train"].map(transform_example, remove_collumns["image_path", batched=False)
test_dataset = dataset["test"].map(transform_example, remove_collumns["image_path"], batched=False)
