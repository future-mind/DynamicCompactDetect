# Tiled Detection Parameter Comparison

This report compares different parameter configurations for tiled detection to help identify optimal settings for tiny object detection.

## Configuration Comparison

| Parameter | Configuration 1 | Configuration 2 |
|-----------|----------------|----------------|
| Tile Size | 320px | 256px |
| Overlap | 20% | 30% |
| Model | YOLOv8n | YOLOv8n |
| Test Images | 2 (zidane.jpg, bus.jpg) | 2 (zidane.jpg, bus.jpg) |

## Performance Metrics Comparison

### Detection Count Improvement

| Configuration | Mean | Median | Min | Max |
|---------------|------|--------|-----|-----|
| 320px, 20% overlap | 14.00 | 14.00 | 8.00 | 20.00 |
| 256px, 30% overlap | 20.00 | 20.00 | 16.00 | 24.00 |

The smaller tile size with higher overlap detected significantly more objects (average of 20 additional objects compared to 14).

### Confidence Score Impact

| Configuration | Mean | Median | Min | Max |
|---------------|------|--------|-----|-----|
| 320px, 20% overlap | -0.0726 | -0.0726 | -0.1074 | -0.0378 |
| 256px, 30% overlap | -0.1970 | -0.1970 | -0.2543 | -0.1396 |

The smaller tile size configuration resulted in a larger decrease in confidence scores, suggesting a higher likelihood of false positives.

### Processing Time Increase

| Configuration | Mean (seconds) | Median (seconds) | Min (seconds) | Max (seconds) |
|---------------|----------------|-----------------|--------------|---------------|
| 320px, 20% overlap | 0.6028 | 0.6028 | 0.5544 | 0.6513 |
| 256px, 30% overlap | 1.2262 | 1.2262 | 1.1253 | 1.3272 |

The smaller tile size with higher overlap approximately doubles the processing time, representing a significant performance trade-off.

### Average Object Size Difference

| Configuration | Mean | Median | Min | Max |
|---------------|------|--------|-----|-----|
| 320px, 20% overlap | -144029.56 | -144029.56 | -212451.82 | -75607.31 |
| 256px, 30% overlap | -161597.58 | -161597.58 | -237238.63 | -85956.54 |

Both configurations detect significantly smaller objects compared to regular detection, with the 256px configuration finding slightly smaller objects on average.

## Analysis

The comparison reveals clear trade-offs between the two parameter configurations:

1. **Detection Capability**: The 256px tile size with 30% overlap detects approximately 43% more objects than the 320px configuration with 20% overlap. This makes it more effective for applications where finding all potential objects is critical.

2. **Confidence Trade-off**: The smaller tile size configuration shows a much larger drop in confidence scores (-0.1970 vs -0.0726), suggesting less certainty in its detections. This could result in more false positives.

3. **Performance Impact**: The smaller tile size configuration requires roughly twice the processing time (1.23s vs 0.60s), making it less suitable for real-time applications.

4. **Object Size**: Both configurations effectively detect smaller objects, with the 256px configuration slightly better at finding the smallest objects.

## Recommendations

Based on the comparison:

1. For **general-purpose applications** with moderate time constraints, the 320px tile size with 20% overlap provides a good balance between improved detection and performance.

2. For **critical detection scenarios** where finding every possible object is important and processing time is less critical, the 256px tile size with 30% overlap is recommended.

3. For **real-time applications**, the processing time increase of both approaches may be prohibitive, but the 320px configuration is significantly more viable.

4. For **tiny object detection**, both configurations show benefits, with the 256px configuration having a slight edge in detecting the smallest objects.

## Next Steps

Further optimization could explore:

1. Testing intermediate configurations (e.g., 288px with 25% overlap)
2. Implementing parallel processing to mitigate the time impact
3. Developing adaptive tiling that applies smaller tiles only in image regions likely to contain tiny objects
4. Combining tiled detection with model ensemble approaches for improved confidence in detections 