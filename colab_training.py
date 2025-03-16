#!/usr/bin/env python3
"""
Google Colab training script for DynamicCompact-Detect.

This script is designed to be run in a Google Colab notebook.
It sets up the environment, downloads the dataset, and trains the model.
"""

# Mount Google Drive (uncomment if needed)
"""
from google.colab import drive
drive.mount('/content/drive')
"""

# Check GPU availability
print("Checking GPU availability...")
!nvidia-smi

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Clone repository
print("\nCloning repository...")
!git clone https://github.com/future-mind/DynamicCompactDetect.git
%cd DynamicCompactDetect

# Install dependencies
print("\nInstalling dependencies...")
!pip install -r requirements.txt

# Download COCO dataset
print("\nDownloading COCO dataset (validation set)...")
!python scripts/download_coco.py --download-val

# Optional: Download training set (uncomment if needed)
# print("\nDownloading COCO dataset (training set)...")
# !python scripts/download_coco.py --download-train

# Start training
print("\nStarting training...")
!python scripts/training/train.py \
    --epochs 10 \
    --batch-size 8 \
    --img-size 640 \
    --name colab_demo

# Test trained model
print("\nTesting trained model...")
!python scripts/inference.py \
    --weights outputs/runs/colab_demo/best_model.pt \
    --source datasets/coco/val2017/000000119445.jpg \
    --conf-thres 0.25

# Run debug analysis 
print("\nRunning debug analysis...")
!python scripts/debug/debug_model.py \
    --weights outputs/runs/colab_demo/best_model.pt \
    --img-path datasets/coco/val2017/000000119445.jpg \
    --conf-thres 0.001

# Save model to Google Drive (uncomment if needed)
"""
print("\nSaving model to Google Drive...")
!mkdir -p /content/drive/MyDrive/DynamicCompact-Models
!cp outputs/runs/colab_demo/best_model.pt /content/drive/MyDrive/DynamicCompact-Models/
print("Model saved to Google Drive at: /content/drive/MyDrive/DynamicCompact-Models/best_model.pt")
"""

print("\nTraining complete!") 