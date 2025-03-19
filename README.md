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

```bash
# Download and prepare the COCO dataset
python data/download_dataset.py
```

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

## Experiment Summary

DynamicCompactDetect achieves the following performance on the COCO validation set:

| Model               | mAP@0.5:0.95 | mAP@0.5 | Inference Speed (FPS) | Model Size (MB) |
|---------------------|--------------|---------|------------------------|-----------------|
| DynamicCompactDetect| TBD          | TBD     | TBD                    | TBD             |
| YOLOv8-n            | 37.3         | 52.9    | 905                    | 6.3             |
| YOLOv8-s            | 44.9         | 61.8    | 428                    | 22.6            |
| YOLOv8-m            | 50.2         | 67.1    | 232                    | 52.2            |
| YOLOv8-l            | 52.9         | 69.8    | 165                    | 87.7            |

*Note: The performance of DynamicCompactDetect will be updated after benchmarking.*

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