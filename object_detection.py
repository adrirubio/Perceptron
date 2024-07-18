import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the COCO dataset
dataset = load_dataset('coco', split={'train': 'train', 'test': 'test'})

# Access the train and test splits
train_dataset = dataset['train']
test_dataset = dataset['test']

# Print the first example from the training dataset
print(train_dataset[0])

# Print the first example from the testing dataset
print(test_dataset[0])
