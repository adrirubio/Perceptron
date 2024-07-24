import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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

# Define paths to the annotation files and images
train_annotation_file = 'path_to_train_annotations.json'
train_image_dir = 'path_to_train_images'
test_annotation_file = 'path_to_test_annotations.json'
test_image_dir = 'path_to_test_images'

# Load datasets
train_dataset = torchvision.datasets.CocoDetection(
    root=train_image_dir,
    annFile=train_annotation_file,
    transform=transformer_train
)

test_dataset = torchvision.datasets.CocoDetection(
    root=test_image_dir,
    annFile=test_annotation_file,
    transform=transformer_test
)

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

# Instanciate the model
num_classes = 91 # 80 classes + 1 background for COCO
model = CNN(num_classes)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define the losses and optimizer
criterion_class = CrossEntropyLoss()
criterion_bbox = nn.MNSLoss()
optimizer = torch.optim.Adam(model.paramaters(), lr=0.001)

# Training loop
def batch_gd(model, criterion_class, criterion_bbox, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train()  # Set model to training mode
        t0 = datetime.now()
        train_loss = []

        for inputs, targets in train_loader:
            # Move data to GPU
            inputs = inputs.to(device)
            labels = targets["annotations"].to(device)
            bbox_targets = targets["bbox"].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            class_logits, bbox_preds = model(inputs)
            loss_class = criterion_class(class_logits, labels)
            loss_bbox = criterion_bbox(bbox_preds, bbox_targets)
            loss = loss_class + loss_bbox

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        # Get train loss
        train_loss = np.mean(train_loss)
        train_losses[it] = train_loss

        model.eval() # Set model to evaluation mode
        test_loss[]
        with torch.no_grad():
          for inputs, targets in test_loader:
            # Move data to the GPU
            inputs = inputs.to(device)
            labels = targets["annotations"].to(device)
            bbox_targets = targets["bbox"].to(device)

            # Forward pass
            class_logits, bbox_preds = model(inputs)
            loss_class = criterion_class(class_logits, labels)
            loss_bbox = criterion_bbox(bbox_preds, bbox_targets)
            loss = loss_class + loss_bbox

            test_loss.append(loss.item())

          # Get test loss
          test_loss = np.mean(test_loss)
          test_losses[it] = test_loss

          dt = datetime.now()

          print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Duration: {dt}')

        return train_losses, test_losses

# Train the model
train_losses, test_losses = batch_gd(
    model, criterion_class, criterion_bbox, optimizer, train_loader, test_loader, epochs=15)
          
plt.plot(train_losses, label="Train loss")
plt.plot(test_losses, label="Test loss")
plt.legend()
plt.show()

# Save model
model_save_path = "object_detection_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
