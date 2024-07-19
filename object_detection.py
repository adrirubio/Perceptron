import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Define transformations for data augmentation and normalization
transformer_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.RandomAffine(0, translate=(0.1, 0.1)),
  transforms.ToTensor(),
])

transformer_test = transforms.Compose([
  transforms.Resize((32, 32)),
  transforms.ToTensor(),
])

# Create function to apply transformations
def transform_example(example, transforms):
  image = Image.open(example["image_path"]).convert("RGB")
  example["image"] = transform(image)
  return example

# Load the COCO dataset
dataset = load_dataset('coco', split={'train': 'train', 'test': 'test'})

# Access the train and test splits
train_dataset = dataset['train']
test_dataset = dataset['test']

# Print the first example from the training dataset
print(train_dataset[0])

# Print the first example from the testing dataset
print(test_dataset[0])
