import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
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

# Load CIFAR-100 training dataset
train_dataset = torchvision.datasets.CIFAR100(
    root='./data',
    train=True,
    transform=transformer_train,
    download=True)

# Load CIFAR-100 test dataset
test_dataset = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=True)

# Define number of classes in our case 100
K = len(set(train_dataset.targets))
print("Number of classes: ", K)

# Data loader
# Useful because it automatically generates batches in the training loop
# and takes care of shuffling
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

for inputs, targets in train_loader:
  print("Inputs: ", inputs)
  print("Targets: ", targets)
  break

# Create DataLoader for training without data augmentation (for accuracy evaluation)
train_dataset_fixed = torchvision.datasets.CIFAR100(
    root=".",
    train=True,
    transform=transforms.ToTensor(),
    download=True)
train_loader_fixed = torch.utils.data.DataLoader(
    dataset=train_dataset_fixed,
    batch_size=batch_size,
    shuffle=False)

# Define the model
class CNN(nn.Module):
  def __init__(self, K):
    super(CNN, self).__init__()
    
    # define the conv layers
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
  self.relu = nn.ReLU() # Define ReLU for use in the dense layer
  self.dropout = nn.Dropout() # Define Dropout for use in the dense layer
  self.fc2 = nn.Linear(1024, K)

def forward(self, x):
  # Forward pass through the convolutional layers
  x = self.conv1(x)
  x = self.conv2(x)
  x = self.conv3(x)

  # Flatten the output for the dense layers
  x = x.view(x.size(0), -1)

  # Forward pass through the dense layers with ReLU and Dropout
  x = self.dropout(x, p=0.5)
  x = self.relu(self.fc1(x))
  x = self.dropout(x, p=0.2)
  x = self.fc2(x)
