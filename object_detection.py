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
def apply_transform(example, transform):
  image = Image.open(example["image_path"]).convert("RGB")
  example["image"] = transform(image)
  return example

# Load the COCO dataset
dataset = load_dataset('coco', split={'train': 'train', 'test': 'test'})

# Access the train and test splits and apply transformations
train_dataset = dataset['train'].map(lambda x: apply_transform(x, transformer_train), batched=False)
test_dataset = dataset['test'].map(lambda x: apply_transform(x, transformer_test), batched=False)

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
                                         shuffle=False)

# Display some examples from the DataLoader
print("Training Batch Example:")
for batch in train_loader:
  print(batch)
  break

print("\nTesting Batch Example:")
for batch in test_loader:
  print(batch)
  break

# Define a simple CNN for object detection
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # Define the convolutional layers
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

        # Define the dense (fully connected) layers for classification
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(1024, num_classes)

        # Define the dense (fully connected) layers for bounding box regression
        self.fc3 = nn.Linear(128 * 4 * 4, 1024)
        self.fc4 = nn.Linear(1024, 4)  # 4 coordinates for bounding box

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the output for the dense layers
        x_flat = x.view(x.size(0), -1)

        # Forward pass through the dense layers for classification
        x_class = self.dropout(x_flat)
        x_class = self.relu(self.fc1(x_class))
        x_class = self.dropout(x_class)
        class_logits = self.fc2(x_class)

        # Forward pass through the dense layers for bounding box regression
        x_bbox = self.dropout(x_flat)
        x_bbox = self.relu(self.fc3(x_bbox))
        x_bbox = self.dropout(x_bbox)
        bbox_coordinates = self.fc4(x_bbox)

        return class_logits, bbox_coordinates
