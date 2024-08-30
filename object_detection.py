import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

transformer_train = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to a fixed size
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Random rotation within 15 degrees
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Apply color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transformer_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define paths to the image and annotation files
train_image_dir = 'coco/train2017'
val_image_dir = 'coco/val2017'
train_annotation_file = 'coco/annotations/instances_train2017.json'
val_annotation_file = 'coco/annotations/instances_val2017.json'

train_dataset = torchvision.datasets.CocoDetection(
    root=train_image_dir,
    annFile=train_annotation_file,
    transform=transformer_train
)

test_dataset = torchvision.datasets.CocoDetection(
    root=val_image_dir,
    annFile=val_annotation_file,
    transform=transformer_test
)

# Print the first example from the training dataset
print(train_dataset[0])

# Print the first example from the testing dataset
print(test_dataset[0])

# Custom collate function to handle variable-size targets and images with no annotations
def collate_fn(batch):
  images = []
  targets = []
  for img, target in batch:
    if len(target) > 0:
      images.append(img)
      targets.append(target)
  if len(images) == 0:
    return torch.empty(0), torch.empty(0)
  return torch.stack(images, 0), targets

# Create data loader
batch_size = 32
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=collate_fn)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         collate_fn=collate_fn)

# Display some examples from the DataLoader
print("Training Batch Example:")
for inputs, targets in train_loader:
  print(inputs.shape)
  print(targets)
  break

print("\n Testing Batch Example:")
for inputs, targets in test_loader:
  print(inputs.shape)
  print(targets)
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
print(device)

# Define the losses and optimizer
criterion_class = nn.CrossEntropyLoss()
criterion_bbox = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop  a bit modified
def batch_gd(model, criterion_class, criterion_bbox, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train()  # Set model to training mode
        t0 = datetime.now()
        train_loss = []

        for inputs, targets in train_loader:
          if inputs.size(0) == 0:
            continue
          # Move data to GPU
          inputs = inputs.to(device)
            
          # Extract annotations
          labels = []
          bbox_targets = []
          for target in targets:
            if len(target) > 0:
              labels.append(target[0]['category_id'])  # Get category ids
              bbox_targets.append(target[0]['bbox'])  # Get bounding boxes
            
          # Convert to tensor
          labels = torch.tensor(labels, dtype=torch.long).to(device)
          bbox_targets = torch.tensor(bbox_targets, dtype=torch.float32).to(device)

          # Forward pass
          class_logits, bbox_preds = model(inputs)
          loss_class = criterion_class(class_logits, labels)
          loss_bbox = criterion_bbox(bbox_preds, bbox_targets)
          loss = loss_class + loss_bbox

          # Backward and optimize
          optimizer.zero_grad()  # Fixed: Added optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          train_loss.append(loss.item())

        # Get train loss
        train_loss = np.mean(train_loss)
        train_losses[it] = train_loss

        model.eval() # Set model to evaluation mode
        test_loss = []
        with torch.no_grad():
          for inputs, targets in test_loader:
            if inputs.size(0) == 0:
              continue

            # Move data to the GPU
            inputs = inputs.to(device)

            # Extract annotations
            labels = []
            bbox_targets = []
            for target in targets:
              if len(target) > 0:
                labels.append(target[0]['category_id'])  # Get category ids
                bbox_targets.append(target[0]['bbox'])  # Get bounding boxes
            
            # Convert to tensor
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            bbox_targets = torch.tensor(bbox_targets, dtype=torch.float32).to(device)

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

# Plot the loss
plt.plot(train_losses, label="Train loss")
plt.plot(test_losses, label="Test loss")
plt.legend()
plt.show()

# Save model
model_save_path = "object_detection_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Accuracy
n_correct = 0
n_total = 0
for inputs, targets in train_loader:
  if inputs.size(0) == 0:
    continue
  
  # Move data to GPU
  inputs = inputs.to(device)

  # Extract annotations
  labels = []
  for target in targets:
    if len(target) > 0:
      labels.append(target[0]['category_id'])  # Get category ids

  # Convert to tensor
  if len(labels) == 0:
    continue  # Skip if no labels are present
  # Convert to tensor
  labels = torch.tensor(labels, dtype=torch.long).to(device)

  # Forward pass
  class_logits, _ = model(inputs)

  # Get predictions
  _, predictions = torch.max(class_logits, 1)

  # Update counts
  n_correct += (predictions == labels).sum().item()
  n_total += labels.size(0)
  
train_acc = n_correct / n_total

n_correct = 0
n_total = 0
for inputs, targets in test_loader:
  if inputs.size(0) == 0:
    continue

  # Move data to the GPU
  inputs = inputs.to(device)

  # Extract annotations
  labels = []
  for target in targets:
    if len(target) > 0:
      labels.append(target[0]['category_id'])  # Get category ids

  # Convert to tensor
  if len(labels) == 0:
    continue  # Skip if no labels are present
  # Convert to tensor
  labels = torch.tensor(labels, dtype=torch.long).to(device)

  # Forward pass
  class_logits, _ = model(inputs)

  # Get predictions
  _, predictions = torch.max(class_logits, 1)

  # Update counts
  n_correct += (predictions == labels).sum().item()
  n_total += labels.size(0)
  
test_acc = n_correct / n_total
print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

def infer_and_display(image_path, model, transform, device):
  # Load the image
  image = Image.open(image_path).convert("RGB")
  
  # Apply transformations
  image_tensor = transform(image).unsqueeze(0).to(device)

  # Perform inference
  with torch.no_grad():
    _, bbox_preds = model(image_tensor)
    bbox_preds = bbox_preds.cpu().squeeze().tolist()

  # Ensure bbox_preds has the correct format
  if len(bbox_preds) != 4:
    print("Error: Bounding box predictions are not in correct format.")
    return
  
  # Convert bbox from [xmin, ymin, width, height] to [xmin, ymin, xmax, ymax]
  bbox = [bbox_preds[0], bbox_preds[1], bbox_preds[0] + bbox_preds[2], bbox_preds[1] + bbox_preds[3]]

  # Draw bounding box on the image
  draw = ImageDraw.Draw(image)
  draw.rectangle(bbox, outline="red", width=3)

  # Display image
  plt.imshow(image)
  plt.axis("off")
  plt.show()

image_path = input("Input your image's path: ")

image = Image.open(image_path).convert("RGB")

print("Original Image:")

# Display the image
plt.imshow(image)
plt.axis('off')  # Optional: turns off the axis
plt.show()

print("Modified Image:")
infer_and_display(image_path, model, transformer_test, device)
