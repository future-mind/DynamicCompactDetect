# DynamicCompactDetect: Comprehensive Usage Guide

This guide provides detailed instructions for using all aspects of the DynamicCompactDetect (DCD) project, from setup to benchmarking, visualization, and research paper generation.

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Running Benchmarks](#2-running-benchmarks)
3. [Visualizing Detections](#3-visualizing-detections)
4. [Generating Reports](#4-generating-reports)
5. [Fine-tuning the Model](#5-fine-tuning-the-model)
6. [Research Paper Data Generation](#6-research-paper-data-generation)
7. [End-to-End Pipeline](#7-end-to-end-pipeline)
8. [Custom Integration](#8-custom-integration)
9. [Troubleshooting](#9-troubleshooting)

## 1. Environment Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/future-mind/dynamiccompactdetect.git
cd dynamiccompactdetect
```

### 1.2 Create a Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 1.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 1.4 Verify Directory Structure

Ensure the following directories exist (they will be created automatically by the scripts if missing):
```bash
mkdir -p models data/test_images results/{benchmarks,comparisons}
```

## 2. Running Benchmarks

The benchmark process evaluates both YOLOv8n and DynamicCompactDetect models on test images, measuring inference time, detection counts, and confidence scores.

### 2.1 Generate Benchmark Data

```bash
python scripts/generate_benchmarks.py
```

This script:
1. Loads both YOLOv8n and DynamicCompactDetect models
2. Runs inference on test images in `data/test_images/`
3. Measures inference time, detection count, and confidence for each model
4. Saves the benchmark results to `results/benchmarks/` as JSON files

**Expected Output:**
- `results/benchmarks/YOLOv8n.json`
- `results/benchmarks/DynamicCompactDetect.json`

### 2.2 Examining Benchmark Results

To view the benchmark results directly:

```bash
cat results/benchmarks/YOLOv8n.json
cat results/benchmarks/DynamicCompactDetect.json
```

Each JSON file contains:
- `inference_time`: Average inference time in milliseconds
- `detection_count`: Average number of objects detected per image
- `confidence`: Average confidence score for detections
- `model_size`: Size of the model in MB

## 3. Visualizing Detections

The comparison script generates visual comparisons of detections from both models.

### 3.1 Run Model Comparison

```bash
python scripts/compare_models.py
```

This script:
1. Loads both YOLOv8n and DynamicCompactDetect models
2. Runs inference on test images
3. Generates side-by-side comparison visualizations
4. Creates performance charts (inference time, detection count, confidence)
5. Saves all visualizations to `results/comparisons/`

**Expected Output:**
- Side-by-side detection images (e.g., `comparison_bus.jpg`)
- Individual detection outputs for each model in separate folders
- Performance charts comparing both models

### 3.2 Examining Visualization Results

Open the generated comparison images in `results/comparisons/` to visually assess detection quality.

## 4. Generating Reports

The report generation script creates a detailed Markdown report comparing both models.

### 4.1 Generate Comparison Report

```bash
python scripts/generate_report.py
```

This script:
1. Loads benchmark data from `results/benchmarks/`
2. Calculates comparative metrics (speed improvement, confidence difference)
3. Generates a comprehensive report with tables and analysis
4. Saves the report to `results/comparisons/model_comparison_report.md`

**Expected Output:**
- `results/comparisons/model_comparison_report.md`

### 4.2 Examining the Report

Open the generated report in a Markdown viewer or directly in your IDE:

```bash
cat results/comparisons/model_comparison_report.md
```

The report includes:
- Performance summary table
- Comparative analysis of key metrics
- Speed improvement calculation
- Detection capability comparison
- Confidence score analysis
- Conclusion on model suitability for edge deployment

## 5. Fine-tuning the Model

The fine-tuning script allows you to adapt DynamicCompactDetect to your custom dataset.

### 5.1 Prepare Your Dataset

Prepare your dataset in YOLOv8 format:
- A YAML file describing your dataset structure
- Images and labels in the appropriate directories

### 5.2 Run Fine-tuning

```bash
python scripts/finetune_dynamiccompactdetect_v8.py \
  --data path/to/your/dataset.yaml \
  --epochs 50 \
  --batch-size 16 \
  --img 640 \
  --output-dir results/finetuned
```

This script:
1. Loads the base DynamicCompactDetect model
2. Configures the training parameters
3. Fine-tunes the model on your dataset
4. Saves the fine-tuned model and training metrics

**Expected Output:**
- Fine-tuned model in the specified output directory
- Training metrics and logs
- Validation results and confusion matrix

### 5.3 Evaluating the Fine-tuned Model

```bash
python scripts/test_dynamiccompactdetect_inference.py \
  --model results/finetuned/weights/best.pt \
  --data data/test_images \
  --output results/finetuned_test
```

## 6. Research Paper Data Generation

The paper data generation script creates comprehensive data for research papers.

### 6.1 Generate Paper Data

```bash
python scripts/generate_paper_data.py
```

This script:
1. Runs comprehensive benchmarks on both models
2. Generates all figures and tables needed for the paper
3. Creates ablation study results
4. Saves all data to `results/research_paper/`

**Expected Output:**
- Tables and figures in appropriate formats
- Comprehensive benchmark data
- Ablation study results

## 7. End-to-End Pipeline

The pipeline script automates the entire workflow from benchmarking to report generation.

### 7.1 Run Complete Pipeline

**macOS/Linux:**
```bash
./run_dcd_pipeline.sh
```

**Windows:**
```bash
powershell -ExecutionPolicy Bypass -File run_dcd_pipeline.ps1
```

This script:
1. Checks for required files and directories
2. Downloads test images if needed
3. Runs model comparison tests
4. Generates comparison visualizations and performance charts
5. Generates the comparison report
6. Optionally generates research paper data

### 7.2 Pipeline Options

```
Usage: ./run_dcd_pipeline.sh [options]

Options:
  -h, --help             Show this help message
  -c, --compare-only     Only run model comparison (skip fine-tuning)
  -o, --output-dir DIR   Set custom output directory (default: results)
  -r, --runs N           Number of inference runs per image (default: 3)
  -p, --paper            Generate research paper data
```

## 8. Custom Integration

### 8.1 Basic Inference in Python

```python
from ultralytics import YOLO

# Load the model
model = YOLO('models/dynamiccompactdetect_finetuned.pt')

# Run inference
results = model('path/to/image.jpg')

# Process results
for r in results:
    print(f"Detected {len(r.boxes)} objects")
    
    # Access individual detections
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates
        conf = box.conf[0]            # Get confidence score
        cls = int(box.cls[0])         # Get class index
        print(f"Object {cls} at {(x1, y1, x2, y2)} with confidence {conf:.2f}")
```

### 8.2 Integration with Frameworks

DynamicCompactDetect can be integrated with any framework that supports YOLO models, including:
- OpenCV
- TensorFlow (via ONNX)
- PyTorch
- ONNX Runtime

Example for ONNX conversion:
```bash
python -c "from ultralytics import YOLO; YOLO('models/dynamiccompactdetect_finetuned.pt').export(format='onnx')"
```

## 9. Troubleshooting

### 9.1 Model Not Found

If you encounter model not found errors:
```
Check that your models are in the correct location:
- YOLOv8n: models/yolov8n.pt
- DynamicCompactDetect: models/dynamiccompactdetect_finetuned.pt

The scripts will attempt to download models automatically if missing.
```

### 9.2 CUDA Issues

If you encounter CUDA-related errors:
```
# Force CPU execution
export CUDA_VISIBLE_DEVICES=""
```

### 9.3 Memory Issues

If you encounter memory issues during inference:
```
# Reduce batch size
python scripts/compare_models.py --batch-size 1

# Use smaller image size
python scripts/compare_models.py --img-size 320
```

### 9.4 Missing Dependencies

If you encounter missing dependency errors, ensure you've installed all requirements:
```bash
pip install -r requirements.txt

# If specific packages are missing
pip install ultralytics opencv-python numpy torch
```

## Conclusion

This guide covers all aspects of using DynamicCompactDetect. For further assistance, refer to the documentation in the `docs/` directory or submit an issue on GitHub. 