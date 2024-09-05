import torch
import torch.nn as nn
import torch.optim as optim

# Test if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
