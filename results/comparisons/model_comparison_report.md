# Model Comparison Report

*Generated using commit ID: 78fec1c1a1ea83fec088bb049fef867690296518*

## Performance Summary

| Model | Inference Time (ms) | Detections | Confidence | Model Size (MB) |
|-------|---------------------|------------|------------|----------------|
| YOLOv8n | 244.94 | 4.5 | 0.667 | 6.23 |
| DynamicCompactDetect | 26.69 | 4.5 | 0.604 | 6.23 |

## Comparative Analysis

- DynamicCompactDetect is **218.25 ms faster** (89.1%) than YOLOv8n
- Both models detect approximately the same number of objects
- YOLOv8n has **9.4% higher confidence** in its detections

- DynamicCompactDetect is **0.00 MB larger** (0.1%) than YOLOv8n

## Conclusion

DynamicCompactDetect demonstrates superior performance in terms of inference speed with a small trade-off in detection confidence. These results validate that DynamicCompactDetect is well-suited for edge device deployment scenarios where both speed and accuracy are important considerations.

## Authors

Abhilash Chadhar and Divya Athya
