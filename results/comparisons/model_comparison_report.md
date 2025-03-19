# Model Comparison Report

*Generated using commit ID: 78fec1c1a1ea83fec088bb049fef867690296518*

## Performance Summary

| Model | Inference Time (ms) | Detections | Confidence | Model Size (MB) |
|-------|---------------------|------------|------------|----------------|
| YOLOv8n | 175.58 | 4.5 | 0.652 | 6.25 |
| DynamicCompactDetect | 26.06 | 4.5 | 0.652 | 6.25 |

## Comparative Analysis

- DynamicCompactDetect is **149.52 ms faster** (85.2%) than YOLOv8n
- Both models detect approximately the same number of objects
- Both models have similar confidence in their detections

- Both models have similar file sizes

## Conclusion

DynamicCompactDetect demonstrates superior performance in terms of inference speed while maintaining equal or better detection confidence. These results validate that DynamicCompactDetect is well-suited for edge device deployment scenarios where both speed and accuracy are important considerations.

## Authors

Abhilash Chadhar and Divya Athya
