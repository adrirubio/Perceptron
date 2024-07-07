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
