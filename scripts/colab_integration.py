#!/usr/bin/env python3
"""
Google Colab integration script for DynamicCompactDetect.

This script provides functions to set up and use DynamicCompactDetect in a Google Colab environment.
It enables:
- Cloning the repository
- Setting up the environment
- Downloading datasets
- Training models
- Running inference

Usage in Colab:
```python
!wget https://raw.githubusercontent.com/future-mind/DynamicCompactDetect/main/scripts/colab_integration.py
import colab_integration as dcd
dcd.setup()  # Clone repo and install dependencies
dcd.download_dataset("coco_val")  # Download dataset
dcd.train(epochs=10, batch_size=8)  # Train model
```
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from IPython.display import display, Markdown, clear_output
from tqdm.notebook import tqdm
import torch

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Repository information
REPO_URL = "https://github.com/future-mind/DynamicCompactDetect.git"
REPO_DIR = "DynamicCompactDetect"

def run_command(cmd, verbose=True):
    """Run a shell command and display output."""
    if verbose:
        print(f"Running: {cmd}")
    
    process = subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    output = []
    for line in iter(process.stdout.readline, ''):
        if not line:
            break
        if verbose:
            print(line.strip())
        output.append(line.strip())
    
    process.wait()
    return process.returncode, output

def display_header(text, level=1):
    """Display a markdown header."""
    header = "#" * level
    display(Markdown(f"{header} {text}"))

def setup(force_fresh=False):
    """
    Set up the DynamicCompactDetect environment in Colab.
    
    Args:
        force_fresh (bool): If True, remove existing repo and clone again
    """
    if not IN_COLAB:
        print("This function is designed to run in Google Colab")
        return
    
    display_header("Setting up DynamicCompactDetect")
    
    # Check for existing repo
    if os.path.exists(REPO_DIR) and not force_fresh:
        print(f"Repository {REPO_DIR} already exists")
        
        # Update the repository
        display_header("Updating repository", level=2)
        os.chdir(REPO_DIR)
        run_command("git pull")
        os.chdir("..")
    else:
        # Remove existing repo if force_fresh
        if os.path.exists(REPO_DIR):
            print(f"Removing existing repository {REPO_DIR}")
            run_command(f"rm -rf {REPO_DIR}")
        
        # Clone the repository
        display_header("Cloning repository", level=2)
        run_command(f"git clone {REPO_URL}")
    
    # Install dependencies
    display_header("Installing dependencies", level=2)
    os.chdir(REPO_DIR)
    run_command("pip install -r requirements.txt")
    
    # Install PyTorch if not installed
    if not hasattr(torch, '__version__'):
        display_header("Installing PyTorch", level=2)
        run_command("pip install torch torchvision")
    
    # Install project in development mode
    run_command("pip install -e .")
    
    # Check GPU availability
    display_header("Checking GPU", level=2)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU is available: {device_name}")
        run_command("nvidia-smi")
    else:
        print("No GPU detected. Training will be slow.")
    
    # Return to home directory
    os.chdir("..")
    
    display_header("Setup Complete", level=2)
    print(f"DynamicCompactDetect is ready to use in {os.path.abspath(REPO_DIR)}")

def download_dataset(dataset_name="coco_val"):
    """
    Download a dataset for training/inference.
    
    Args:
        dataset_name (str): Name of dataset to download
                           Options: "coco_val", "coco_train", "coco_full"
    """
    if not IN_COLAB:
        print("This function is designed to run in Google Colab")
        return
    
    if not os.path.exists(REPO_DIR):
        print(f"Repository not found. Run setup() first.")
        return
    
    os.chdir(REPO_DIR)
    
    display_header(f"Downloading {dataset_name} dataset")
    
    if dataset_name == "coco_val":
        run_command("python scripts/download_coco.py --download-val")
    elif dataset_name == "coco_train":
        run_command("python scripts/download_coco.py --download-train")
    elif dataset_name == "coco_full":
        run_command("python scripts/download_coco.py --download-val --download-train")
    else:
        print(f"Unknown dataset: {dataset_name}")
        print("Available options: coco_val, coco_train, coco_full")
    
    os.chdir("..")

def train(epochs=10, batch_size=8, img_size=640, model_name="colab_model", 
          weights=None, device=""):
    """
    Train the DynamicCompact model.
    
    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Input image size
        model_name (str): Name for the training run
        weights (str): Path to initial weights or None for scratch
        device (str): Device to train on ('' for auto-detect)
    """
    if not IN_COLAB:
        print("This function is designed to run in Google Colab")
        return
    
    if not os.path.exists(REPO_DIR):
        print(f"Repository not found. Run setup() first.")
        return
    
    os.chdir(REPO_DIR)
    
    display_header(f"Training DynamicCompact model")
    print(f"Training with: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
    
    # Check dataset exists
    if not os.path.exists("datasets/coco/val2017"):
        print("Dataset not found. Downloading COCO validation set...")
        run_command("python scripts/download_coco.py --download-val")
    
    # Construct training command
    cmd = [
        "python scripts/training/train.py",
        f"--epochs {epochs}",
        f"--batch-size {batch_size}",
        f"--img-size {img_size}",
        f"--output runs/train_{model_name}"
    ]
    
    if weights:
        cmd.append(f"--weights {weights}")
    
    if device:
        cmd.append(f"--device {device}")
    
    # Run training
    display_header("Starting training", level=2)
    run_command(" ".join(cmd))
    
    # Verify training output
    output_dir = f"runs/train_{model_name}"
    if os.path.exists(f"{output_dir}/best_model.pt"):
        print(f"Training complete! Model saved to {output_dir}/best_model.pt")
    else:
        print("Training may not have completed successfully.")
    
    os.chdir("..")

def run_inference(weights, source, conf_thres=0.25, img_size=640, save_dir=None):
    """
    Run inference using a trained model.
    
    Args:
        weights (str): Path to model weights
        source (str): Path to image, video, or directory
        conf_thres (float): Confidence threshold for detections
        img_size (int): Input image size
        save_dir (str): Output directory or None for default
    """
    if not IN_COLAB:
        print("This function is designed to run in Google Colab")
        return
    
    if not os.path.exists(REPO_DIR):
        print(f"Repository not found. Run setup() first.")
        return
    
    os.chdir(REPO_DIR)
    
    display_header(f"Running inference")
    
    # Check if weights file exists
    if not os.path.exists(weights):
        print(f"Weights file not found: {weights}")
        return
    
    # Construct inference command
    cmd = [
        "python scripts/inference.py",
        f"--weights {weights}",
        f"--source {source}",
        f"--conf-thres {conf_thres}",
        f"--img-size {img_size}"
    ]
    
    if save_dir:
        cmd.append(f"--output {save_dir}")
    
    # Run inference
    display_header("Processing images", level=2)
    run_command(" ".join(cmd))
    
    os.chdir("..")

def mount_drive():
    """Mount Google Drive in Colab."""
    if not IN_COLAB:
        print("This function is designed to run in Google Colab")
        return
    
    display_header("Mounting Google Drive")
    
    from google.colab import drive
    drive.mount('/content/drive')
    
    print("Google Drive mounted at /content/drive")
    return "/content/drive/MyDrive"

def save_model_to_drive(model_path, drive_dir="DynamicCompactDetect"):
    """
    Save a trained model to Google Drive.
    
    Args:
        model_path (str): Path to model file
        drive_dir (str): Directory in Google Drive to save to
    """
    if not IN_COLAB:
        print("This function is designed to run in Google Colab")
        return
    
    # Mount drive if not already mounted
    if not os.path.exists('/content/drive'):
        drive_path = mount_drive()
    else:
        drive_path = "/content/drive/MyDrive"
    
    display_header(f"Saving model to Google Drive")
    
    # Check if model exists
    if not os.path.exists(model_path):
        full_path = f"{REPO_DIR}/{model_path}"
        if not os.path.exists(full_path):
            print(f"Model not found at {model_path} or {full_path}")
            return
        model_path = full_path
    
    # Create directory in Google Drive
    save_dir = f"{drive_path}/{drive_dir}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get model filename
    model_name = os.path.basename(model_path)
    save_path = f"{save_dir}/{model_name}"
    
    # Copy model to Drive
    print(f"Copying {model_path} to {save_path}")
    import shutil
    shutil.copy2(model_path, save_path)
    
    print(f"Model saved to Google Drive at: {save_path}")

def visualize_results(images_dir):
    """
    Visualize detection results from the output directory.
    
    Args:
        images_dir (str): Directory containing result images
    """
    if not IN_COLAB:
        print("This function is designed to run in Google Colab")
        return
    
    from IPython.display import Image, display
    import glob
    import random
    
    display_header("Visualization of Detection Results")
    
    # Find result images
    if os.path.exists(REPO_DIR):
        full_path = f"{REPO_DIR}/{images_dir}"
    else:
        full_path = images_dir
    
    if not os.path.exists(full_path):
        print(f"Results directory not found: {full_path}")
        return
    
    image_files = glob.glob(f"{full_path}/*.jpg") + glob.glob(f"{full_path}/*.png")
    
    if not image_files:
        print(f"No images found in {full_path}")
        return
    
    # Display a sample of images (up to 5)
    display_header("Sample Detection Results", level=2)
    
    sample = min(5, len(image_files))
    for img_path in random.sample(image_files, sample):
        display(Image(img_path))
        print(f"Image: {os.path.basename(img_path)}")

# Main execution for quick demo
if __name__ == "__main__" and IN_COLAB:
    display(Markdown("""
    # DynamicCompactDetect in Colab
    
    This script provides helper functions to use DynamicCompactDetect in Google Colab.
    
    ## Available Functions:
    - `setup()` - Clone repo and install dependencies
    - `download_dataset()` - Download COCO dataset
    - `train()` - Train a model
    - `run_inference()` - Run inference on images
    - `mount_drive()` - Mount Google Drive
    - `save_model_to_drive()` - Save model to Google Drive
    - `visualize_results()` - Display detection results
    
    ## Example Usage:
    ```python
    # Setup
    setup()
    
    # Download dataset
    download_dataset("coco_val")
    
    # Train for 10 epochs
    train(epochs=10, batch_size=8)
    
    # Run inference
    run_inference(
        weights="runs/train_colab_model/best_model.pt", 
        source="datasets/coco/val2017/000000119445.jpg"
    )
    
    # Save to Drive
    mount_drive()
    save_model_to_drive("runs/train_colab_model/best_model.pt")
    ```
    """))
    
    # Print system info
    print("\nSystem Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}") 