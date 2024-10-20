import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import io
import os

# Use CUDA_LAUNCH_BLOCKING for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Load the SST-2 dataset
dataset = load_dataset("glue", "sst2")

# Split into train and test sets
train_dataset = dataset['train']
test_dataset = dataset['test']

# Display some examples from the train set
print("Training Examples:")
for i in range(3):
    print(f"Input: {train_dataset[i]['sentence']}")
    print(f"Target: {train_dataset[i]['label']}")

# Display some examples from the test set
print("Testing Examples:")
for i in range(3):
    print(f"Input: {test_dataset[i]['sentence']}")
    print(f"Target: {test_dataset[i]['label']}")

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and encode the datasets
train_encodings = tokenizer(train_dataset["sentence"], truncation=True, padding=True, max_length=128)
train_labels = torch.tensor(train_dataset["label"]).long()  # Ensure labels are in correct format

test_encodings = tokenizer(test_dataset["sentence"], truncation=True, padding=True, max_length=128)
test_labels = torch.tensor(test_dataset["label"]).long()  # Ensure labels are in correct format

# Convert to PyTorch tensors
train_input_ids = torch.tensor(train_encodings["input_ids"])
train_attention_masks = torch.tensor(train_encodings["attention_mask"])

test_input_ids = torch.tensor(test_encodings["input_ids"])
test_attention_masks = torch.tensor(test_encodings["attention_mask"])

# Create tensor datasets
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

# Data loader
# Useful because it automatically generates batches in the training loop
# and takes care of shuffling

batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Display one example from each DataLoader
print("Training Batch Example:")
for batch in train_loader:
    print(batch)
    break  # Just to display one batch

print("\nTesting Batch Example:")
for batch in test_loader:
    print(batch)
    break  # Just to display one batch

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Mode model to the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        model.train()  # Set model to training mode
        t0 = datetime.now()
        train_loss = []

        for inputs, masks, targets in train_loader:
            # Move data to GPU
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass with error handling
            try:
                outputs = model(inputs, attention_mask=masks)
                logits = outputs.logits  # Extract the logits
                loss = criterion(logits, targets)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
            except RuntimeError as e:
                print(f"Error during training iteration: {e}")
                continue

        # Get train loss
        train_loss = np.mean(train_loss)

        model.eval()
        test_loss = []

        with torch.no_grad():
            for inputs, masks, targets in test_loader:
                # Move data to the GPU
                inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

                # Forward pass with error handling
                try:
                    outputs = model(inputs, attention_mask=masks)
                    logits = outputs.logits  # Extract the logits
                    loss = criterion(logits, targets)

                    test_loss.append(loss.item())
                except RuntimeError as e:
                    print(f"Error during testing iteration: {e}")
                    continue

        # Get test loss
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        dt = datetime.now() - t0

        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, Duration: {dt}')

    return train_losses, test_losses

# Train the model
train_losses, test_losses = batch_gd(
    model, criterion, optimizer, train_loader, test_loader, epochs=5)

# Plot the train and test loss
plt.plot(train_losses, label="train_loss")
plt.plot(test_losses, label="test_loss")
plt.legend()
plt.show()

# Save model
model_save_path = "/home/adrian/Documents/Perceptron/model_weights/sentiment_analisis_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Save the tokenizer
tokenizer_save_path = "/home/adrian/Documents/Perceptron/model_weights/sentiment_analysis_tokenizer"
tokenizer.save_pretrained(tokenizer_save_path)
print(f"Tokenizer saved to {tokenizer_save_path}")

# Accuracy
model.eval()
n_correct = 0
n_total = 0

for inputs, masks, targets in train_loader:
    inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

    # Forward pass
    outputs = model(inputs, attention_mask=masks)

    # Extract the logits
    logits = outputs.logits

    # Get predictions
    _, predictions = torch.max(logits, 1)

    # Update counts
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

train_acc = n_correct / n_total

n_correct = 0
n_total = 0

for inputs, masks, targets in test_loader:
    inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

    # Forward pass
    outputs = model(inputs, attention_mask=masks)

    # Extract the logits
    logits = outputs.logits

    # Get prediction
    _, predictions = torch.max(logits, 1)

    # Update counts
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

test_acc = n_correct / n_total
print(f"Train acc: {train_acc}, Test acc {test_acc}")

# Function to predict the sentiment of new sentences
def predict_sentences(model, tokenizer, sentences):
    model.eval()
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions

# Test with a few random sentences
example_sentences = [
    "This movie was fantastic!",
    "I hated every moment of this film.",
    "It was an average experience, nothing special.",
    "The plot was very predictable and boring."
]

predictions = predict_sentences(model, tokenizer, example_sentences)
for sentence, prediction in zip(example_sentences, predictions):
    print(f"Sentence: {sentence} => Prediction: {'Positive' if prediction.item() == 1 else 'Negative'}")
