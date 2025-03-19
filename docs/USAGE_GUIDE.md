# DynamicCompactDetect: Comprehensive Usage Guide

This document provides detailed instructions for using the DynamicCompactDetect (DCD) model, including environment setup, running benchmarks, performing inference, and fine-tuning the model.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Available Scripts](#available-scripts)
3. [Running Benchmarks](#running-benchmarks)
4. [Comparing Models](#comparing-models)
5. [Visualizing Detections](#visualizing-detections)
6. [Fine-tuning on Custom Data](#fine-tuning-on-custom-data)
7. [Generating Research Paper Data](#generating-research-paper-data)
8. [Running the Complete Pipeline](#running-the-complete-pipeline)
9. [Troubleshooting](#troubleshooting)
10. [Additional Resources](#additional-resources)

## Environment Setup

### Prerequisites
- Python 3.8 or later
- CUDA-compatible GPU (optional but recommended for training)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/future-mind/dynamiccompactdetect.git
   cd dynamiccompactdetect
   ```

2. Create and activate a virtual environment:
   
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

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation:
   ```bash
   python -c "from ultralytics import YOLO; print('Installation successful!')"
   ```

## Available Scripts

DynamicCompactDetect comes with several utility scripts:

| Script | Description |
|--------|-------------|
| `compare_models.py` | Compare inference performance between YOLOv8n and DynamicCompactDetect |
| `generate_benchmarks.py` | Generate detailed benchmark data for both models |
| `generate_report.py` | Create a comprehensive comparison report |
| `compare_all_models.py` | Compare across multiple model variants (if available) |
| `finetune_dynamiccompactdetect_v8.py` | Fine-tune the model on custom datasets |
| `generate_paper_data.py` | Generate data and figures for research paper |
| `generate_visualizations.py` | Create visualizations of model performance |
| `test_dynamiccompactdetect_inference.py` | Simple test script for inference |

## Running Benchmarks

To evaluate the performance of DynamicCompactDetect and compare it with YOLOv8n:

```bash
# Generate benchmarks for both models
python scripts/generate_benchmarks.py
```

This script:
1. Loads both YOLOv8n and DynamicCompactDetect models
2. Runs inference on test images in `data/test_images/`
3. Measures inference time, detection count, and confidence scores
4. Saves benchmark data to `results/benchmarks/`

### Command-line Arguments

```
--output-dir OUTPUT_DIR   Directory to save benchmark results (default: results/benchmarks)
--num-runs NUM_RUNS       Number of inference runs for more accurate timing (default: 5)
--edge-device             Enable edge device simulation (slower but more realistic for edge deployment)
```

Example with custom settings:
```bash
python scripts/generate_benchmarks.py --num-runs 10 --edge-device
```

### Output

The script saves JSON files with benchmark results:
- `results/benchmarks/YOLOv8n.json`
- `results/benchmarks/DynamicCompactDetect.json`

Each file contains:
- `model_name`: Name of the model
- `model_path`: Path to the model file
- `inference_time`: Average inference time in milliseconds
- `detection_count`: Average number of detections per image
- `confidence`: Average confidence score for detections
- `num_images`: Number of test images used
- `images_with_detections`: Number of images where objects were detected

## Comparing Models

For a detailed comparison between YOLOv8n and DynamicCompactDetect:

```bash
python scripts/compare_models.py
```

This script:
1. Runs inference with both models on test images
2. Generates side-by-side comparison images
3. Creates performance charts
4. Generates a comprehensive comparison report

### Command-line Arguments

```
--num-runs NUM_RUNS       Number of inference runs for timing (default: 5)
--output-dir OUTPUT_DIR   Directory to save comparison results (default: results/comparisons)
--edge-device             Enable edge device simulation (CPU-only, single thread)
```

Example with custom settings:
```bash
python scripts/compare_models.py --num-runs 25 --edge-device
```

### Output

The script generates:
1. Comparison images showing detections from both models side-by-side
   - `results/comparisons/comparison_*.png`
2. Performance comparison chart
   - `results/comparisons/performance_comparison.png`
3. Markdown report summarizing the comparison
   - `results/comparisons/model_comparison_report.md`

## Visualizing Detections

To visualize detections from a specific model:

```bash
python scripts/test_dynamiccompactdetect_inference.py --model dcd --image data/test_images/bus.jpg --save-output
```

### Command-line Arguments

```
--model MODEL             Model to use for inference ('yolov8n' or 'dcd') (default: dcd)
--image IMAGE             Path to test image (default: data/test_images/bus.jpg)
--num-runs NUM_RUNS       Number of inference runs for timing (default: 5)
--output-dir OUTPUT_DIR   Directory to save results (default: results)
--save-output             Save output images with detections
```

## Fine-tuning on Custom Data

To fine-tune DynamicCompactDetect on your custom dataset:

```bash
python scripts/finetune_dynamiccompactdetect_v8.py --data path/to/your/dataset.yaml --epochs 50
```

### Dataset Preparation

1. Organize your dataset in YOLO format:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   │   ├── img1.jpg
   │   │   └── ...
   │   └── labels/
   │       ├── img1.txt
   │       └── ...
   ├── val/
   │   ├── images/
   │   │   ├── img_val1.jpg
   │   │   └── ...
   │   └── labels/
   │       ├── img_val1.txt
   │       └── ...
   └── data.yaml
   ```

2. Create a YAML file (`data.yaml`) with dataset configuration:
   ```yaml
   path: ./dataset  # Root directory
   train: train/images  # Train images relative to path
   val: val/images  # Validation images relative to path

   # Class names
   names:
     0: person
     1: car
     # Add your classes here
   ```

### Command-line Arguments

```
--data DATA               Path to dataset YAML file
--epochs EPOCHS           Number of training epochs (default: 100)
--batch-size BATCH_SIZE   Batch size for training (default: 16)
--img IMG                 Input image size (default: 640)
--device DEVICE           Device for training ('cpu', '0', '0,1,2,3', etc.) (default: GPU if available)
--weights WEIGHTS         Initial weights path (default: models/dynamiccompactdetect_finetuned.pt)
--name NAME               Experiment name (default: finetune_dcd)
```

### Training Monitoring

The script creates a `runs/` directory with training logs, which you can monitor using TensorBoard:

```bash
# Install TensorBoard if not already installed
pip install tensorboard

# Launch TensorBoard
tensorboard --logdir runs/
```

## Generating Research Paper Data

To generate data and figures for the research paper:

```bash
python scripts/generate_paper_data.py
```

This script:
1. Runs comprehensive benchmarks on both models
2. Generates tables and figures for the research paper
3. Saves results to `results/research_paper/`

### Command-line Arguments

```
--output-dir OUTPUT_DIR   Output directory for benchmark results (default: results)
```

## Running the Complete Pipeline

For a complete end-to-end run of all components:

```bash
# Unix-like systems (macOS/Linux)
./run_dcd_pipeline.sh

# Windows (PowerShell)
./run_dcd_pipeline.ps1
```

This script:
1. Checks for required files and directories
2. Downloads test images if needed
3. Runs model comparison tests
4. Generates benchmark data
5. Creates visualization and comparison reports
6. Outputs a comprehensive report to `results/` directory

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'ultralytics'**
   - Solution: Make sure you've installed all dependencies with `pip install -r requirements.txt`

2. **CUDA out of memory**
   - Solution: Reduce batch size with `--batch-size 4` or use CPU with `--device cpu`

3. **No such file or directory: 'models/dynamiccompactdetect_finetuned.pt'**
   - Solution: Run comparison scripts which will download the model automatically or manually download it

4. **File 'data/test_images/bus.jpg' not found**
   - Solution: Create the data directory and download test images:
     ```bash
     mkdir -p data/test_images
     wget https://github.com/ultralytics/assets/raw/main/bus.jpg -O data/test_images/bus.jpg
     wget https://github.com/ultralytics/assets/raw/main/zidane.jpg -O data/test_images/zidane.jpg
     ```

5. **Different benchmark results than in the paper**
   - Solution: Hardware variations can impact benchmark results. Use `--edge-device` for consistent results

### Getting Help

If you encounter issues not covered in this guide:
1. Check the GitHub repository issues section
2. Consult the Ultralytics YOLOv8 documentation for model-specific issues
3. Ensure you're using compatible hardware and software versions

## Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Research Paper](docs/research_paper/DynamicCompactDetect_Paper.md)
- [Model Architecture Details](docs/research_paper/ARCHITECTURE.md) 