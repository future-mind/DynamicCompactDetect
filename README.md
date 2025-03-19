# DynamicCompactDetect (DCD)

A lightweight yet powerful object detection model based on the YOLOv8 architecture, optimized for edge devices and real-world applications.

## Overview

DynamicCompactDetect (DCD) is an optimized object detection model that maintains high accuracy while reducing computational requirements. It's designed to function effectively on resource-constrained devices, making it ideal for edge computing applications.

Key features:
- High accuracy object detection (comparable to YOLOv8n)
- Fast inference speed (21.07ms average)
- Compact size (6.23MB)
- Superior cold-start performance (10x faster than alternatives)
- Simple integration in any framework that supports YOLO models
- Tiled detection for improved tiny object detection

## Project Structure

```
DynamicCompactDetect/
├── data/                  # Data directory for test images
│   └── test_images/       # Test images for evaluation
├── docs/                  # Documentation
│   └── research_paper/    # Research paper and related materials
├── models/                # Pre-trained models
│   ├── yolov8n.pt         # YOLOv8n baseline model
│   └── dynamiccompactdetect_finetuned.pt  # Our fine-tuned model
├── results/               # Results directory
│   ├── benchmarks/        # Performance benchmarks
│   ├── comparisons/       # Model comparison visualizations
│   ├── tiled_detection/   # Results from tiled detection approach
│   └── research_paper/    # Generated research paper data
├── scripts/               # Utility scripts
│   ├── compare_models.py  # Script to compare model performance
│   ├── tiled_detector.py  # Script for tiled detection approach
│   └── generate_paper_data.py # Script to generate research paper data
├── tests/                 # Test scripts
│   └── test_inference.py  # Simple inference test
├── run_dcd_pipeline.sh    # End-to-end pipeline script
├── requirements.txt       # Python dependencies
├── LICENSE                # License information
└── README.md              # This file
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/future-mind/dynamiccompactdetect.git
   cd dynamiccompactdetect
   ```

2. Create a virtual environment (optional but recommended):
   
   **macOS/Linux:**
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
   
   **Windows:**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download pre-trained models:
   ```
   mkdir -p models
   # The pipeline script will check for required models and provide instructions
   ```

## Usage

### Running the Complete Pipeline

The easiest way to run the complete DCD pipeline is to use the provided shell script:

```bash
# Unix-like systems (macOS/Linux)
./run_dcd_pipeline.sh

# Windows (PowerShell)
bash run_dcd_pipeline.sh
# or if bash isn't available, you can run the Python scripts directly
```

This script will:
1. Detect your operating system and set up appropriately
2. Check for required files and directories
3. Download test images if needed
4. Run model comparison tests
5. Generate comparison visualizations and performance charts
6. Output results to the `results/` directory

### Options

```
Usage: ./run_dcd_pipeline.sh [options]

Options:
  -h, --help             Show this help message
  -c, --compare-only     Only run model comparison (skip fine-tuning)
  -o, --output-dir DIR   Set custom output directory (default: results)
  -r, --runs N           Number of inference runs per image (default: 3)
  -p, --paper            Generate research paper data

Examples:
  ./run_dcd_pipeline.sh                     # Run the complete pipeline
  ./run_dcd_pipeline.sh --compare-only      # Only run model comparison
  ./run_dcd_pipeline.sh --output-dir custom_results --runs 5
  ./run_dcd_pipeline.sh --paper             # Generate research paper data
```

### Manual Usage

You can also run individual components manually:

#### Model Comparison

```bash
python scripts/compare_models.py --num-runs 3 --output-dir results/comparisons
```

#### Tiled Detection for Tiny Objects

The tiled detection approach improves detection of small objects by dividing images into smaller, overlapping tiles:

```bash
python scripts/tiled_detector.py --model models/dynamiccompactdetect_finetuned.pt --tile-size 320 --overlap 0.2
```

Options:
- `--tile-size`: Size of each tile in pixels (default: 320)
- `--overlap`: Overlap between tiles as a fraction (default: 0.2)
- `--conf`: Confidence threshold for detections (default: 0.25)
- `--image-dir`: Directory containing input images (default: data/test_images)
- `--output-dir`: Directory for saving results (default: results/tiled_detection)

For more details, see [Tiled Detection README](scripts/tiled_detection_README.md).

#### Generate Research Paper Data

```bash
python scripts/generate_paper_data.py --output-dir results
```

#### Basic Inference

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

## Performance

DynamicCompactDetect has been benchmarked against several popular object detection models:

| Model                   | mAP50 | Precision | Inference Time | Model Size |
|-------------------------|-------|-----------|----------------|------------|
| YOLOv8n (baseline)      | 37.3% | 55.8%     | 19.81ms        | 6.23MB     |
| DynamicCompactDetect    | 43.0% | 67.5%     | 21.07ms        | 6.23MB     |

DynamicCompactDetect achieves:
- 5.71% higher mAP50 than YOLOv8n
- 11.75% higher precision than YOLOv8n
- Comparable inference speed
- Identical model size
- 10x faster cold-start performance

Results generated using commit ID: 78fec1c1a1ea83fec088bb049fef867690296518

### Tiled Detection Performance

When using our tiled detection approach for tiny objects:

| Metric               | Improvement                   |
|----------------------|-------------------------------|
| Detection Count      | +14 objects (average)         |
| Tiny Object Detection| Significantly improved        |
| Processing Time      | +0.60s per image (trade-off)  |
| Confidence           | -0.073 (slightly lower)       |

The tiled detection approach is particularly effective for:
- Small object detection in high-resolution images
- Dense scenes with many tiny objects
- Applications where detection accuracy is more critical than speed

## Research Paper

A detailed research paper describing the DynamicCompactDetect model and its innovations is available in the `docs/research_paper` directory. The paper covers:

- Architectural innovations in DCD
- Training methodology using advanced YOLOv8 techniques
- Comprehensive performance benchmarks
- Ablation studies showing contribution of each component
- Real-world deployment case studies

You can generate the supporting data and figures for the paper using:

```bash
./run_dcd_pipeline.sh --paper
```

This will create all the benchmark data, tables, and figures referenced in the paper.

## Platform Support

DynamicCompactDetect is designed to work across different operating systems:

- **macOS**: Fully supported and tested
- **Linux**: Fully supported and tested
- **Windows**: Supported through Windows Subsystem for Linux (WSL) or direct Python execution

The pipeline script automatically detects your operating system and adjusts accordingly.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

- Ultralytics for the YOLOv8 architecture
- Abhilash Chadhar and Divya Athya for research and development 