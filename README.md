# DynamicCompact-Detect

A lightweight and efficient object detection model optimized for real-time performance.

## Overview

DynamicCompactDetect is a novel object detection framework that balances efficiency and accuracy for resource-constrained environments. The model is designed to be dynamic and adaptable, with a compact architecture that enables real-time detection on various hardware platforms.

## Features

- Lightweight architecture with only 454,686 parameters
- Fast inference speed suitable for real-time applications
- YOLO-style detection approach
- Support for COCO dataset (80 classes)
- Configurable settings for different speed/accuracy tradeoffs

## Model Comparison

This repository includes comparison scripts to evaluate DynamicCompactDetect against other state-of-the-art models:
- YOLOv10
- RT-DETR

## Directory Structure

```
DynamicCompactDetect/
├── src/                       # Core source code
│   ├── models/                # Model definitions
│   ├── utils/                 # Utility functions
│   └── data/                  # Data processing code
├── scripts/                   # Executable scripts
│   ├── training/              # Training scripts
│   ├── evaluation/            # Evaluation scripts
│   ├── comparison/            # Model comparison scripts
│   └── debug/                 # Debugging utilities
├── configs/                   # Configuration files
│   ├── model_configs/         # Model architecture configs
│   └── train_configs/         # Training hyperparameters
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Unit tests
├── docs/                      # Documentation
└── outputs/                   # Output directory (gitignored)
    ├── runs/                  # Training runs
    ├── checkpoints/           # Model checkpoints
    └── results/               # Evaluation results
```

## Installation

```bash
# Clone the repository
git clone https://github.com/future-mind/DynamicCompactDetect.git
cd DynamicCompactDetect

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Dataset Setup

For COCO dataset:

```bash
# Download and setup COCO dataset
python scripts/download_coco.py
```

## Training

```bash
# Basic training
python scripts/training/train.py

# With custom parameters
python scripts/training/train.py --batch-size 16 --epochs 100 --img-size 640
```

## Inference

```bash
# Run inference on sample images
python scripts/inference.py --weights outputs/checkpoints/best_model.pt --source path/to/image.jpg

# With lower confidence threshold for debugging
python scripts/inference.py --weights outputs/checkpoints/best_model.pt --source path/to/image.jpg --conf-thres 0.001
```

## Model Comparison

```bash
# Compare with other detection models
python scripts/comparison/compare_models.py --models dynamiccompact,yolov10,rtdetr
```

## License

MIT License

## Citation

If you use this code in your research, please cite:

```
@article{future-mind2023dynamiccompact,
  title={DynamicCompactDetect: A Lightweight Object Detection Framework for Resource-Constrained Environments},
  author={Future Mind},
  year={2023}
}
```

## Contact

For questions or support, please open an issue on GitHub or contact us directly at contact@future-mind.io. 