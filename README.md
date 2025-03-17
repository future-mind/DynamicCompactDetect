# DynamicCompactDetect

A lightweight and efficient object detection model based on YOLO architecture, optimized for both speed and accuracy.

## Overview

DynamicCompactDetect is a compact object detection model designed for real-time applications. It features:

- Efficient backbone architecture with optimized layers
- Dynamic head for better feature utilization
- Improved loss function for more accurate detections
- Support for Exponential Moving Average (EMA) during training
- Comprehensive training, testing, and comparison scripts

## Installation

```bash
# Clone the repository
git clone https://github.com/future-mind/DynamicCompactDetect.git
cd DynamicCompactDetect

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Training

Train the model on the COCO dataset:

```bash
python scripts/train_dynamiccompactdetect.py --data coco.yaml --epochs 100 --batch-size 16 --device 0
```

For a quick test on a small dataset:

```bash
python scripts/train_dynamiccompactdetect.py --data coco8.yaml --epochs 3 --batch-size 1 --device cpu
```

### Inference

Run inference using a trained model:

```bash
python scripts/test_dynamiccompactdetect_inference.py --weights runs/train/dynamiccompactdetect/weights/best.pt --source data/samples
```

### Model Comparison

Compare the performance of the original and fine-tuned models:

```bash
python scripts/compare_dynamiccompactdetect_models.py --original-model dynamiccompactdetect.pt --fine-tuned-model runs/train/dynamiccompactdetect/weights/best.pt
```

## Repository Structure

```
DynamicCompactDetect/
├── data/                  # Dataset configurations and sample images
├── scripts/               # Training, inference, and utility scripts
│   ├── train_dynamiccompactdetect.py       # Training script
│   ├── test_dynamiccompactdetect_inference.py  # Inference script
│   └── compare_dynamiccompactdetect_models.py  # Model comparison script
├── src/                   # Source code
│   ├── model.py           # Model architecture definition
│   ├── utils/             # Utility functions
│   │   ├── general.py     # General utility functions
│   │   ├── loss.py        # Loss functions
│   │   └── plots.py       # Visualization utilities
├── weights/               # Pre-trained model weights
└── requirements.txt       # Dependencies
```

## Performance

DynamicCompactDetect achieves competitive performance on the COCO dataset:

- mAP50: 0.861
- mAP50-95: 0.619
- Inference time: ~20ms per image on CPU

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This model builds upon the YOLO family of models and incorporates insights from various research papers in the field of object detection. 