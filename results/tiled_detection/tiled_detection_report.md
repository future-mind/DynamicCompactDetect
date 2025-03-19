# Tiled Detection Performance Report

## Overview
This report summarizes the performance of the tiled detection approach compared to regular detection using YOLOv8n on test images. The tiled detection method splits images into overlapping tiles, processes each tile separately, and then merges the results.

## Test Configuration
- **Model**: YOLOv8n
- **Tile Size**: 320px
- **Overlap**: 20%
- **Test Images**: 2 (zidane.jpg, bus.jpg)

## Performance Metrics

### Detection Count
- **Mean Increase**: 14.00 additional detections
- **Median Increase**: 14.00 additional detections
- **Range**: 8.00 to 20.00 additional detections

This indicates that tiled detection consistently finds more objects than regular detection, likely because breaking the image into smaller tiles helps the model identify objects that might be missed when processing the entire image at once.

### Confidence Scores
- **Mean Difference**: -0.0726 (lower in tiled detection)
- **Median Difference**: -0.0726
- **Range**: -0.1074 to -0.0378

The lower confidence scores in tiled detection suggest that while more objects are detected, the model is less certain about these additional detections. This could indicate potential false positives or that the model is detecting partially visible objects at tile boundaries.

### Processing Time
- **Mean Increase**: 0.6028 seconds
- **Median Increase**: 0.6028 seconds
- **Range**: 0.5544 to 0.6513 seconds

Tiled detection takes significantly longer than regular detection, which is expected due to the overhead of processing multiple tiles and merging results.

### Average Object Size
- **Mean Difference**: -144029.56 (smaller in tiled detection)
- **Range**: -212451.82 to -75607.31

The objects detected by tiled detection are generally smaller in pixel area, suggesting that the approach is successfully identifying smaller objects that regular detection might miss.

## Sample Results
The visual comparisons in `zidane_comparison.jpg` and `bus_comparison.jpg` show the differences in detection between the two methods.

## Conclusion
The tiled detection approach significantly increases the number of detected objects at the cost of processing time and potentially lower confidence in detections. This trade-off may be worthwhile for applications where detecting small objects is critical and processing time is not a constraint.

## Recommendations
1. Consider using tiled detection for datasets with many tiny objects
2. Experiment with different tile sizes and overlap percentages to optimize the performance
3. Implement confidence thresholding to filter out potential false positives
4. For time-sensitive applications, consider parallel processing of tiles to reduce overall detection time 