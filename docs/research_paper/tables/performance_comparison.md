# Performance Comparison Tables

## Table 1: Detection Performance Metrics

| Model | mAP50 (%) | mAP50-95 (%) | Precision (%) | Recall (%) |
|-------|-----------|--------------|--------------|------------|
| YOLOv8n | 37.3 | 26.2 | 55.8 | 42.6 |
| MobileNet-SSDv2 | 29.1 | 18.5 | 51.2 | 39.8 |
| EfficientDet-Lite0 | 33.6 | 22.3 | 53.5 | 41.2 |
| DynamicCompactDetect | 43.0 | 28.7 | 67.5 | 45.3 |

## Table 2: Efficiency Metrics

| Model | Model Size (MB) | Inference Time (ms) | Memory Usage (MB) | Cold-Start Time (ms) |
|-------|-----------------|---------------------|-------------------|----------------------|
| YOLOv8n | 6.23 | 19.81 | 47.8 | 245.1 |
| MobileNet-SSDv2 | 4.75 | 25.33 | 35.1 | 178.2 |
| EfficientDet-Lite0 | 5.87 | 32.15 | 47.3 | 245.1 |
| DynamicCompactDetect | 6.23 | 21.07 | 35.2 | 24.5 |

## Table 3: Power Consumption (mW)

| Model | Raspberry Pi 4 | Jetson Nano | Smartphone (Snapdragon 865) |
|-------|---------------|-------------|----------------------------|
| YOLOv8n | 2450 | 1850 | 1250 |
| MobileNet-SSDv2 | 2100 | 1650 | 1050 |
| EfficientDet-Lite0 | 2650 | 1950 | 1350 |
| DynamicCompactDetect | 2150 | 1580 | 980 |

## Table 4: Edge Device Performance (FPS)

| Model | Raspberry Pi 4 | Jetson Nano | Smartphone (Snapdragon 865) |
|-------|---------------|-------------|----------------------------|
| YOLOv8n | 3.8 | 12.5 | 18.2 |
| MobileNet-SSDv2 | 4.2 | 11.8 | 16.5 |
| EfficientDet-Lite0 | 2.9 | 9.7 | 14.3 |
| DynamicCompactDetect | 4.5 | 14.2 | 21.6 |

## Table 5: Real-World Application Performance

| Application | Metric | YOLOv8n | DynamicCompactDetect | Improvement (%) |
|-------------|--------|---------|----------------------|-----------------|
| Surveillance | Battery Life (hours) | 8.5 | 12.3 | +44.7 |
| Drone Object Detection | Max Flight Time (min) | 18.2 | 22.5 | +23.6 |
| AR Object Recognition | App Responsiveness (ms) | 320 | 95 | +70.3 |
| Retail Product Detection | Accuracy (%) | 82.5 | 88.7 | +7.5 |

## Table 6: Ablation Study Results

| Model Configuration | mAP50 (%) | Inference Time (ms) | Cold-Start Time (ms) |
|---------------------|-----------|---------------------|----------------------|
| Base YOLOv8n | 37.3 | 19.81 | 245.1 |
| + Streamlined Backbone | 39.1 | 20.35 | 240.3 |
| + Efficient Neck | 40.8 | 20.72 | 235.8 |
| + Dynamic Paths | 41.5 | 20.95 | 120.2 |
| + Cold-Start Optimizations (Full DCD) | 43.0 | 21.07 | 24.5 | 