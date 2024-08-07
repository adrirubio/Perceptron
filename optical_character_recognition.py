import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

# Load the IAM Handwriting dataset
dataset = load_dataset("Teklia/IAM-line")

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
  # Directly apply the transformations since the image is already loaded
  image = example["image"].convert("RGB")
  example["image"] = transform_train(image)
  return example

# Function to transform images and prepare data for the test set
def transform_test_func(example):
  # Directly apply the transformations since the image is already loaded
  image = example["image"].convert("RGB")
  example["image"] = transform_test(image)
  return example

train_dataset = dataset["train"].map(transform_train_func, batched=False)
test_dataset = dataset["test"].map(transform_test_func, batched=False)

# Encode text labels to numerical indices
label_encoder = LabelEncoder()

# Fit label encoder on the train dataset's text labels
train_texts = [example["text"] for example in train_dataset]
label_encoder.fit(train_texts)

# Define a function to encode text labels to indices
def encode_labels(example):
    example["text"] = label_encoder.transform([example["text"]])[0]
    return example

# Apply label encoding to the datasets
train_dataset = train_dataset.map(encode_labels, batched=False)
test_dataset = test_dataset.map(encode_labels, batched=False)

# Check the first example in the training dataset
train_example = train_dataset[0]
print("Train example image shape: ", train_example["image"])
print("Train example transcription: ", train_example["text"])

# Check the first example in the testing dataset
test_example = test_dataset[0]
print("Test example image shape: ", test_example["image"])
print("Test example transcription: ", test_example["text"])

# Data loader
batch_size = 64
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=True)

# Display some examples from the train_loader
print("Training batch examples")
for batch in train_loader:
    inputs, targets = batch["image"], batch["text"]
    print(inputs)
    print(targets)
    break

# Display some examples from the test_loader
print("\nTesting batch examples")
for batch in test_loader:
    inputs, targets = batch["image"], batch["text"]
    print(inputs)
    print(targets)
    break

class CNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super(CNN, self).__init__()
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
num_classes = len(label_encoder.classes_)  # This should be the number of unique characters or classes in your dataset
model = CNN(num_classes)

# Move data to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train() # Set model to training mode
        t0 = datetime.now()
        train_loss = []

        for batch in train_loader:
            # Get inputs and targets from the batch
            inputs = batch["image"]
            targets = batch["text"]

            # Move data to the GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss) # Compute mean train loss for this epoch
