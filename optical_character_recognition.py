import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the IAM Handwriting dataset
dataset = load_dataset("iam_dataset")
