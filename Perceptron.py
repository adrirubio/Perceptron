import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load DailyDialog dataset
dataset = load_dataset('daily_dialog')

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set padding token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate responses
def generate_response(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
prompt = "Hello, what are you up to?"
if dataset:
    response = generate_response(prompt, model, tokenizer)
    print("Question:", response)

# Access the dialog sections
train_dialog = dataset['train']['dialog']
test_dialog = dataset['test']['dialog']

# Create custom Dataset class
class DailyDialogDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length=512):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = " ".join(self.dialogues[idx])
        encoded = self.tokenizer(dialogue, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        return input_ids, input_ids  # Returning input_ids as both input and target

# Create train and test dataset instances
train_dataset = DailyDialogDataset(train_dialog, tokenizer)
test_dataset = DailyDialogDataset(test_dialog, tokenizer)


# Create batches for optimizing computational efficiency
batch_size = 4
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
)

# Defining the model by freezing the weights
for param in model.transformer.wte.parameters():
  param.requires_grad = False
for layer in model.transformer.h[:-1]:
  for param in layer.parameters():
    param.requires_grad = False

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
  # Arrays to store the loss values for each epoch
  train_losses = np.zeros(epochs)
  test_losses = np.zeros(epochs)

  for it in range(epochs):
    # Set datetime
    t0 = datetime.now()
    train_loss = []
    for inputs, targets in train_loader:
      # Move data to GPU if available
      inputs, targets = inputs.to(device), targets.to(device)

      # Zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)
      loss = criterion(outputs, targets)

      # Backward and optimize
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())

    # Get train loss and test loss
    train_loss = np.mean(train_loss) # a little misleading

    test_loss = []
    for inputs, targets in test_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      test_loss.append(loss.item())
    test_loss = np.mean(test_loss)

    # Save losses
    train_losses[it] = train_loss
    test_losses[it] = test_loss

    dt = datetime.now() - t0
    print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
      Test Loss: {test_loss:.4f}, Duration: {dt}')

  return train_losses, test_losses
