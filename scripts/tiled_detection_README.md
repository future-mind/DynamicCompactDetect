# Tiled Detection for Tiny Object Detection

This module implements a tiled detection approach for improving the detection of small objects in images. By dividing the input image into smaller, overlapping tiles, processing each tile separately, and then merging the results, this method can significantly increase the detection of tiny objects that might be missed by regular detection approaches.

## Installation

Make sure you have the required dependencies installed:

```bash
pip install ultralytics opencv-python numpy matplotlib pillow
```

## Usage

The main script `tiled_detector.py` provides a command-line interface for running tiled detection on images.

### Basic Usage

```bash
python scripts/tiled_detector.py --model models/yolov8n.pt --image-dir data/test_images --output-dir results/tiled_detection
```

### Command Line Options

- `--model`: Path to the YOLOv8 model file (default: models/yolov8n.pt)
- `--tile-size`: Size of each tile in pixels (default: 320)
- `--overlap`: Overlap between tiles as a fraction (default: 0.2)
- `--conf`: Confidence threshold for detections (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.7)
- `--image-dir`: Directory containing input images (default: data/test_images)
- `--output-dir`: Directory for saving results (default: results/tiled_detection)
- `--tiny-threshold`: Size threshold (in pixels) for considering an object as tiny (default: 32)
- `--help`: Show help message and exit

## How It Works

1. **Image Tiling**: The input image is divided into overlapping tiles of specified size.
2. **Tile Processing**: Each tile is processed by the object detection model.
3. **Detection Merging**: The detections from all tiles are combined using Non-Maximum Suppression.
4. **Visualization**: Results are visualized by comparing tiled detection with regular detection.
5. **Statistics**: Performance metrics are calculated and saved.

## Output

The script generates the following outputs in the specified output directory:

- `*_comparison.jpg`: Side-by-side comparison of regular and tiled detection for each input image
- `summary_statistics.txt`: Statistical summary of performance metrics
- `summary_plots.png`: Visual plots comparing key metrics
- `tiled_detection_report.md`: Detailed report of the performance and findings

## Example

```bash
python scripts/tiled_detector.py --model models/yolov8n.pt --tile-size 320 --overlap 0.2 --image-dir data/test_images --output-dir results/tiled_detection
```

## Performance Considerations

Tiled detection typically offers these trade-offs:

- **Pros**: Significantly increased detection of small objects
- **Cons**: Longer processing time, potentially more false positives

For optimal results, experiment with different tile sizes and overlap values based on your specific use case and the typical size of objects in your images.

## Integration

To integrate tiled detection into your own Python code:

```python
from ultralytics import YOLO
import cv2
from scripts.tiled_detector import create_tiles, process_tiles, merge_detections

# Load model
model = YOLO('models/yolov8n.pt')

# Load image
image = cv2.imread('path/to/image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create tiles
tiles, coords = create_tiles(image_rgb, tile_size=320, overlap=0.2)

# Process tiles
tile_results = process_tiles(model, tiles)

# Merge detections
merged_boxes, merged_conf, merged_cls = merge_detections(
    tile_results, coords, image_rgb.shape, iou_threshold=0.7
)

# Now use merged_boxes, merged_conf, merged_cls for your application
``` 