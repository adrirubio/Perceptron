import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2'
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the trained model weights
model_save_path = "gpt2_dailydialog.pt"
model.load_state_dict(torch.load(model_save_path))
model.eval()
