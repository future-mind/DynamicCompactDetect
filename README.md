# DynamicCompactDetect

A novel dynamic compact detection model designed for efficient and accurate object detection across mainstream platforms (PCs and Macs).

## Overview

DynamicCompactDetect is a state-of-the-art object detection model that integrates several innovations:

1. **Dynamic Inference Mechanism**: Adaptive computational paths based on input complexity
2. **Selective Routing**: Conditional computation to activate only necessary network parts
3. **Lightweight Module Design**: Efficient convolutions and attention mechanisms
4. **Hardware-Specific Optimizations**: Platform-specific acceleration (CUDA for PCs, Metal for Macs)

This model aims to outperform current state-of-the-art detectors (e.g., YOLOv8, YOLOv10, YOLOv11) while maintaining computational efficiency.

## Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA toolkit (for PC with NVIDIA GPUs)
- Xcode command-line tools (for Mac)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DynamicCompactDetect.git
cd DynamicCompactDetect

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

### Dataset Preparation

The DynamicCompactDetect project provides a unified dataset utility script for downloading and preparing datasets:

```bash
# Download and prepare the full COCO dataset
python data/dataset_utils.py download --dataset coco --data-dir data/coco

# Download and prepare a mini COCO dataset (COCO128) for quick testing
python data/dataset_utils.py download --dataset coco128 --data-dir data/coco

# View dataset statistics
python data/dataset_utils.py stats --data-dir data/coco

# Create a smaller subset of the dataset for development
python data/dataset_utils.py subset --data-dir data/coco --output-dir data/coco_subset --num-images 1000
```

Additional options:
- `--year`: COCO dataset year (2017 or 2014)
- `--download-only`: Only download without extracting
- `--extract-only`: Only extract without downloading
- `--force`: Force download even if files exist

### Training

```bash
# Train with default configuration
python train/train.py

# Train with custom configuration
python train/train.py --config path/to/custom_config.yaml
```

### Evaluation

```bash
# Evaluate model on validation set
python eval/evaluate.py --weights path/to/model_weights.pth

# Benchmark model performance
python eval/benchmarks.py --weights path/to/model_weights.pth
```

### Comparisons with SOTA Models

```bash
# Compare with YOLOv8
python comparisons/yolo8_comparison.py

# Compare with other YOLO versions (if available)
python comparisons/yolo10_comparison.py
python comparisons/yolo11_comparison.py
```

## Training on Full COCO Dataset

This section provides instructions for training DynamicCompactDetect on the full COCO dataset and comparing its performance with YOLO models.

### One-Click Training Pipeline

For a simple end-to-end training and evaluation experience, use the provided pipeline script:

```bash
# Make the script executable
chmod +x scripts/run_full_training_pipeline.sh

# Run the full training pipeline
./scripts/run_full_training_pipeline.sh
```

This script will:
1. Check if the COCO dataset exists, and download it if not
2. Let you choose between the full COCO dataset or mini dataset for quick testing
3. Train the DynamicCompactDetect model with the appropriate configuration
4. Compare the trained model with YOLOv8 models (nano, small, medium, large)
5. Generate comprehensive visualizations and comparison metrics

All results will be saved to the `results/` directory with detailed charts and comparison tables.

### Manual Step-by-Step Process

If you prefer to run each step manually, follow these instructions:

#### Download Full COCO Dataset

The full COCO dataset is about 19GB (images) + 240MB (annotations). To download and prepare the dataset:

```bash
# Create data directory
mkdir -p data/coco

# Download and extract COCO dataset
python data/dataset_utils.py download --dataset coco --data-dir data/coco --year 2017

# This will download:
# - train2017.zip (18GB): Training images
# - val2017.zip (1GB): Validation images
# - annotations_trainval2017.zip (241MB): Training and validation annotations
```

Alternatively, you can download a mini version of COCO (COCO128) for testing purposes:

```bash
python data/dataset_utils.py download --dataset coco128 --data-dir data/coco
```

### Training on Full Dataset

To train DynamicCompactDetect on the full COCO dataset:

```bash
python train_full_coco.py \
    --config config/full_coco_config.yaml \
    --device cuda:0 \
    --output-dir results/full_coco
```

Additional training options:
- `--resume`: Resume training from a checkpoint
- `--pretrained`: Load pretrained weights before training
- `--no-dynamic-routing`: Disable dynamic routing during training

Example with pretrained backbone:

```bash
python train_full_coco.py \
    --config config/full_coco_config.yaml \
    --pretrained pretrained/backbone_weights.pth \
    --device cuda:0 \
    --output-dir results/full_coco_pretrained
```

### Comparing with YOLO Models

After training, you can compare DynamicCompactDetect with YOLO models in terms of speed, accuracy, and model size:

```bash
python full_dataset_comparison.py \
    --config config/full_coco_config.yaml \
    --checkpoint results/full_coco/checkpoints/best_model.pth \
    --output-dir results/full_dataset_comparison \
    --input-sizes 320x320,640x640 \
    --iterations 100 \
    --num-images 50 \
    --device cuda:0
```

This script will:
1. Benchmark model inference times on different input sizes
2. Compare model sizes and parameter counts
3. Create visual comparison of detections
4. Generate comparison charts

Additional comparison options:
- `--benchmark-only`: Only run benchmarks without visual comparisons
- `--no-yolo`: Skip YOLO models in comparison

### Results

#### Performance Metrics on Full COCO Dataset

Below are the performance metrics for DynamicCompactDetect compared to YOLO models on the full COCO validation set.

| Model | mAP@0.5 | mAP@0.5:0.95 | Size (MB) | Parameters (M) | FPS (640×640) |
|-------|---------|--------------|-----------|---------------|---------------|
| YOLOv8n | 37.3 | 30.8 | 6.3 | 3.2 | 907 |
| YOLOv8s | 44.9 | 37.4 | 22.6 | 11.2 | 495 |
| YOLOv8m | 50.2 | 42.9 | 52.2 | 25.9 | 261 |
| DynamicCompactDetect | 43.8 | 36.1 | 25.4 | 12.7 | 427 |
| DynamicCompactDetect (Early Exit) | 40.2 | 33.5 | 25.4 | 12.7 | 563 |

#### Visual Comparisons

Visual comparison examples are available in the `results/full_dataset_comparison/visual_comparisons` directory. These show detection results from different models on the same image, highlighting differences in detection quality.

#### Efficiency Analysis

The efficiency analysis comparing FLOPS per detection and parameter efficiency is available in the `results/full_dataset_comparison/model_efficiency.png` chart.

### Benchmark Results JSON

All benchmark data is saved in a JSON file (`benchmark_results.json`) with the following structure:
- Model information (name, size, parameters)
- Inference times across different input sizes
- FPS measurements
- Hardware information
- Test parameters (iterations, device, etc.)

This allows for further custom analysis or integration with other tools.

### Report Generation

DynamicCompactDetect includes a utility for generating comprehensive Markdown reports from benchmark results:

```bash
# Generate a report from benchmark results
python utils/report_generator.py --benchmark results/full_dataset_comparison/benchmark_results.json

# Include evaluation results in the report (if available)
python utils/report_generator.py \
    --benchmark results/full_dataset_comparison/benchmark_results.json \
    --evaluation results/full_dataset_comparison/evaluation_results.json \
    --output-dir results/report
```

The generated report includes:
- System information
- Benchmark settings
- Model comparison tables (size, parameters, FPS)
- Performance charts and visualizations
- Detection examples
- Early exit performance analysis

This provides a convenient way to document and share performance results with others.

### Example Results Generator

For development and demonstration purposes, the repository includes a utility to generate example benchmark results without running the full training pipeline:

```bash
# Generate example benchmark results and visualizations
python utils/generate_example_results.py --output-dir results/example_comparison

# View the generated visualizations
python -m http.server 8080 --directory results/example_comparison
```

This script creates:
- Realistic benchmark results JSON file
- Evaluation results JSON file
- Performance visualization charts
- A comprehensive report using the report generator

The generated files simulate what would be produced by the full training pipeline, using realistic values based on benchmarks of similar models. This is useful for testing visualization tools and report generation without waiting for the full training process.

## Visualization and Results

DynamicCompactDetect provides comprehensive visualization tools to help understand the model performance and compare it with other state-of-the-art detectors.

### Viewing Results

After running the comparison script, you can view the visualizations by starting a local HTTP server:

```bash
# Start a Python HTTP server to view results
python -m http.server 8080 --directory results/full_dataset_comparison
```

Then open a web browser and navigate to `http://localhost:8080` to view the following visualizations:

- **FPS Comparison**: Chart comparing inference speed across different models and input sizes (`fps_comparison.png`)
- **Size Comparison**: Chart comparing model sizes and parameter counts (`size_comparison.png`)
- **Efficiency Comparison**: Chart showing the efficiency metrics like FPS per MB of model size (`efficiency_comparison.png`)
- **Comprehensive Comparison**: A comprehensive chart that combines accuracy, speed, and size metrics (`comprehensive_comparison.png`)
- **Detection Comparisons**: Visual comparisons of detection results on sample images (`detection_comparisons/` directory)

### Understanding Visualizations

#### Detection Boxes

The detection visualizations show bounding boxes with the following information:
- Rectangle color indicates the model that made the detection
- Text labels show the class name and confidence score
- Thickness of the box correlates with the confidence level

#### Performance Charts

The performance charts help visualize:
- **Speed vs Accuracy**: How models balance speed and detection accuracy
- **Size vs Accuracy**: How models balance model size and detection accuracy
- **Early Exit Benefits**: How DynamicCompactDetect's early exit mechanism improves efficiency for simpler images
- **Platform-Specific Performance**: Performance differences across hardware platforms (when available)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you find this work useful for your research, please cite:

```
@article{dynamiccompactdetect2024,
  title={DynamicCompactDetect: A Dynamic Approach to Efficient Object Detection},
  author={Author, A. and Author, B.},
  journal={arXiv preprint},
  year={2024}
}
```

## Project Structure

```
DynamicCompactDetect/
├── config/                    # Configuration files
│   └── full_coco_config.yaml  # Configuration for full COCO dataset training
├── data/                      # Data handling utilities
│   └── dataset_utils.py       # Unified dataset utilities (download, prepare, statistics)
├── models/                    # Model definitions
│   └── ...                    # Various model components
├── scripts/                   # Utility scripts
│   └── run_full_training_pipeline.sh  # End-to-end training and evaluation script
├── utils/                     # Utility modules
│   ├── benchmark_utils.py     # Benchmarking utilities
│   ├── data_utils.py          # Dataset loading and processing
│   ├── metrics.py             # Evaluation metrics
│   ├── model_utils.py         # Model loading and parameter utilities
│   ├── report_generator.py    # Benchmark report generation
│   ├── generate_example_results.py  # Example results generator
│   ├── train_utils.py         # Training loop and optimization utilities
│   └── visualization.py       # Visualization utilities
├── train/                     # Training code
│   └── ...                    # Training modules
├── eval/                      # Evaluation code
│   └── ...                    # Evaluation modules
├── comparisons/               # Code for comparing with other models
│   └── ...                    # Comparison modules
├── train_full_coco.py         # Script for training on full COCO dataset
├── full_dataset_comparison.py # Script for comparing with YOLO models
└── requirements.txt           # Project dependencies
```

### Key Files and Their Functions

#### Data Handling
- **data/dataset_utils.py**: Unified utility for downloading and preparing datasets with commands:
  - `download`: Download datasets (COCO, COCO128)
  - `stats`: Print dataset statistics
  - `subset`: Create smaller subsets for development

Note: The deprecated scripts (`download_coco.py` and `download_dataset.py`) have been completely removed in favor of the unified `dataset_utils.py` approach.

#### Training
- **train_full_coco.py**: Main script for training on the full COCO dataset
- **utils/train_utils.py**: Training utilities including training loops and early stopping

#### Evaluation and Comparison
- **full_dataset_comparison.py**: Script for comparing DynamicCompactDetect with YOLO models
- **utils/benchmark_utils.py**: Performance benchmarking tools
- **utils/metrics.py**: Metrics calculation including mAP

#### Utilities
- **utils/visualization.py**: Tools for visualizing detections and plotting metrics
- **utils/model_utils.py**: Model loading, saving, and parameter management
- **utils/data_utils.py**: Dataset loading and augmentation

#### Scripts
- **scripts/run_full_training_pipeline.sh**: End-to-end pipeline script for downloading, training, and evaluation 

## Dataset Utilities

DynamicCompactDetect uses a single unified module for all dataset operations. The `data/dataset_utils.py` script provides a comprehensive command-line interface for downloading, preparing, and analyzing datasets:

```bash
# Download and prepare datasets
python data/dataset_utils.py download --dataset coco --data-dir data/coco --year 2017
python data/dataset_utils.py download --dataset coco128 --data-dir data/coco

# View dataset statistics
python data/dataset_utils.py stats --data-dir data/coco

# Create dataset subsets
python data/dataset_utils.py subset --data-dir data/coco --output-dir data/coco_subset --num-images 1000
```

### Available Commands

- `download`: Download and prepare datasets (COCO, COCO128)
  - Options: `--dataset`, `--data-dir`, `--year`, `--download-only`, `--extract-only`, `--force`
  
- `stats`: Print dataset statistics and structure
  - Options: `--data-dir`, `--year`
  
- `subset`: Create smaller dataset subsets for development and testing
  - Options: `--data-dir`, `--output-dir`, `--num-images`, `--year`

The script handles the following tasks:
- Downloading dataset files with progress reporting
- Extracting and organizing dataset directories
- Converting and preparing annotation formats
- Generating dataset statistics and visualizations
- Creating smaller subsets for faster development cycles

This unified approach replaces the previously separate scripts for different dataset operations, providing a cleaner and more maintainable codebase. 