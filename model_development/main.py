# model_development/main.py
import os
import torch

print("🚀 Starting VGG19 Deepfake Detection Training")
print("=" * 50)

# Test PyTorch installation
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Import and run training
from train import main

if __name__ == "__main__":
    main()