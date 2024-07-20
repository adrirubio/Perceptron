import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
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
def transform(example, transform):
  image = Image.open(example["image_path"]).convert("RGB")
  example["image"] = transform(image)
  return example

# Load the COCO dataset
dataset = load_dataset('coco', split={'train': 'train', 'test': 'test'})

# Access the train and test splits and apply transformations
train_dataset = dataset['train'].map(lambda x: transform(x, transformer_train), batched=False)
test_dataset = dataset['test'].map(lambda x: transform(x, transformer_test), batched=False)

# Set the train and test sets format to PyTorch tensors
train_dataset.set_format(type="torch", columns=["image", "annotations"])
test_dataset.set_format(type="torch", columns=["image", "annotations"])

# Print the first example from the training dataset
print(train_dataset[0])

# Print the first example from the testing dataset
print(test_dataset[0])

# Create data loader
batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         suffle=False)
