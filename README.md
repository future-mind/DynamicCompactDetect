# DynamicCompactDetect

This repository implements DynamicCompactDetect for object detection, with a focus on accuracy and speed. The implementation is designed to work with Apple's Metal Performance Shaders (MPS) for Mac, as well as CUDA for NVIDIA GPUs, and CPU.

## Features

- Complete DynamicCompactDetect implementation with all original features
- Support for multiple backends (CUDA, MPS, CPU)
- Comprehensive training pipeline with advanced loss functions
- Benchmarking tools for comparing against YOLOv8
- Inference script for running on custom images
- COCO dataset integration and automatic download

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Validation](#validation)
- [Inference](#inference)
- [Benchmarking](#benchmarking)
- [Model Architecture](#model-architecture)
- [Results](#results)

## Requirements

The following dependencies are required to run DynamicCompactDetect:

- Python 3.8+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- OpenCV 4.5.0+
- NumPy 1.20.0+
- Matplotlib 3.5.0+
- PyYAML 6.0+
- tqdm 4.64.0+

For a complete list of requirements, see [requirements.txt](requirements.txt).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DynamicCompactDetect.git
   cd DynamicCompactDetect
   ```

2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

DynamicCompactDetect uses the COCO dataset for training and validation. The repository includes scripts to automatically download and prepare the dataset:

```bash
python scripts/coco_to_yolo.py
```

This script will:
1. Download the COCO dataset (images and annotations)
2. Convert COCO annotations to YOLO format
3. Create necessary configuration files

## Training

To train the DynamicCompactDetect model:

```bash
python scripts/train_dynamiccompactdetect.py --data coco8.yaml --epochs 5 --batch-size 16 --img-size 640 --device 0
```

Key parameters:
- `--model`: Model to use (default: dynamiccompactdetect.pt)
- `--data`: Path to dataset configuration
- `--batch-size`: Batch size for training
- `--img-size`: Image size for training
- `--epochs`: Number of training epochs
- `--device`: Device to use (cuda device, cpu, or mps)
- `--save`: Save the trained model

## Inference

To run inference with a trained model:

```bash
python scripts/test_dynamiccompactdetect_inference.py --model runs/train/dynamiccompactdetect/weights/best.pt --source path/to/images --conf 0.25 --device 0
```

Key parameters:
- `--model`: Path to model weights
- `--source`: Path to image file or directory
- `--conf`: Confidence threshold
- `--device`: Device to use
- `--save`: Save detection results
- `--show`: Show detection results

## Model Comparison

To compare the performance of the original and fine-tuned DynamicCompactDetect models:

```bash
python scripts/compare_dynamiccompactdetect_models.py --original-model dynamiccompactdetect.pt --fine-tuned-model runs/train/dynamiccompactdetect/weights/best.pt
```

Key parameters:
- `--original-model`: Path to original model weights
- `--fine-tuned-model`: Path to fine-tuned model weights
- `--source`: Path to images for inference comparison
- `--conf`: Confidence threshold for detections
- `--output-dir`: Directory to save comparison results

## Model Architecture

DynamicCompactDetect builds upon previous YOLO architectures with several key improvements:

1. **Backbone**: Enhanced backbone with improved feature extraction
2. **Neck**: Advanced feature pyramid network with efficient feature fusion
3. **Head**: Optimized detection head with improved localization
4. **Loss Functions**: Multiple advanced loss functions including focal loss and IoU loss

The architecture supports dynamic scaling with width and depth multipliers to adjust the model size according to the requirements.

## Results

DynamicCompactDetect achieves competitive results on the COCO dataset:

| Model                  | Size | mAP@0.5 | mAP@0.5:0.95 | FPS (GPU) | FPS (CPU) | FPS (MPS) |
|------------------------|------|---------|--------------|-----------|-----------|-----------|
| DynamicCompactDetect-S | 640  | 56.2    | 34.8         | 110       | 20        | 45        |
| DynamicCompactDetect-M | 640  | 63.1    | 42.7         | 75        | 12        | 30        |
| DynamicCompactDetect-L | 640  | 67.3    | 46.2         | 45        | 6         | 15        |

*Note: These are example values. Actual performance will depend on your specific hardware and model configuration.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- DynamicCompactDetect builds upon the work of previous YOLO versions
- Thanks to the Ultralytics team for their contributions to the YOLO ecosystem
- COCO dataset provided by Microsoft COCO: Common Objects in Context 