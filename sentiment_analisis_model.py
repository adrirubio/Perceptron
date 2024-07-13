import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import io

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

# Tokenize and encode the train_dataset
train_encodings = tokenizer(train_dataset["sentence"], truncation=True, padding=True, max_length=128)
train_labels = train_dataset["label"]

# Tokenize and encode the test_dataset
test_encodings = tokenizer(test_dataset["sentence"], truncation=True, padding=True, max_length=128)
test_labels = test_dataset["label"]

# Convert to PyTorch tensors
train_input_ids = torch.tensor(train_encodings["input_ids"])
train_attention_mask = torch.tensor(train_encodings["attention_mask"])
train_labels = torch.tensor(train_labels)

test_input_ids = torch.tensor(test_encodings["input_ids"])
test_attention_mask = torch.tensor(test_encodings["attention_mask"])
test_labels = torch.tensor(test_labels)

# Create tensor datasets
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

# Data loader
# Useful because it automatically generates batches in the training loop
# and takes care of shuffling

batch_size = 128
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
    break  # Just to display one batch

print("\nTesting Batch Example:")
for batch in test_loader:
    print(batch)
    break  # Just to display one batch
