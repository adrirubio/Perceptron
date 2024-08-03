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
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to transform images and prepare data
def transform_example(example):
    image = Image.open(example["image_path").convert('RGB')
    example["image"] = transform(image)
    return example

train_dataset = dataset["train"].map(transform_example)
test_dataset = dataset["test"].map(transform_example)
