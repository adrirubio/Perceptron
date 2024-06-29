import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load DailyDialog dataset
dataset = load_dataset('daily_dialog')

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
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

# Access the train and test splits
train_dataset = dataset['train']
test_dataset = dataset['test']

# Example of the train split
print(train_dataset[0])

# Define the dialog section
train_dialog = train_dataset['dialog']
test_dialog = test_dataset['dialog']

# Example of the train_dialog set
print(train_dialog[5])

# Create batches for optimizing computational efficiency
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
)

# Define the model by freezing the weights
for param in model.transformer.wte.parameters():
  param.requires_grad = False
