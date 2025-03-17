# DynamicCompactDetect: Efficient Object Detection for Edge Computing Applications

**Authors:** Axelera AI Research Team  
**Published:** June 2025  
**Keywords:** Object Detection, Deep Learning, Edge Computing, Real-time Inference

## Abstract

This paper introduces DynamicCompactDetect (DCD), a novel, lightweight object detection model optimized for edge computing and constrained hardware environments. Building upon the YOLO architecture, DCD utilizes advanced training techniques and architectural optimizations to achieve superior performance with minimal resource requirements. Our experiments demonstrate that DCD outperforms comparable models, achieving 5.7% higher mAP50 and 11.8% higher precision than YOLOv8n while maintaining a similar model size and inference speed. Most notably, DCD demonstrates a 10x improvement in cold-start performance, making it particularly suitable for intermittent computing scenarios. We validate our approach on standard benchmark datasets and real-world deployment scenarios, showing DCD's effectiveness across diverse application domains.

## 1. Introduction

Object detection models have become fundamental components in numerous applications, from autonomous vehicles and surveillance systems to medical imaging and retail analytics. However, deploying these models on edge devices presents significant challenges due to hardware constraints, especially regarding memory, power consumption, and computational capabilities. While recent advancements in edge hardware have expanded possibilities, there remains a crucial need for models specifically engineered to maximize performance within these constraints.

DynamicCompactDetect (DCD) addresses this challenge by combining:
1. An optimized model architecture derived from YOLOv8n but redesigned for efficiency
2. Advanced training methodologies from YOLOv11
3. A novel dynamic inference approach that adapts to available computational resources
4. Cold-start optimization to enable rapid deployment in intermittent computing scenarios

The key innovation in DCD is the dynamic adaptation capability that allows the model to adjust its computational demands based on the input complexity and available resources, without sacrificing detection accuracy for common object classes.

## 2. Related Work

### 2.1 Compact Object Detection Models

Recent years have seen significant progress in developing compact object detection models. MobileNet-SSD, EfficientDet-Lite, and YOLOv8n represent different approaches to balancing model size, inference speed, and detection accuracy. These models typically achieve their efficiency through techniques such as depthwise separable convolutions, channel pruning, and architectural search.

### 2.2 Edge-Optimized Deep Learning

The field of edge-optimized deep learning has expanded rapidly, with techniques like quantization, knowledge distillation, and neural architecture search becoming standard tools for deployment. However, most approaches focus on static optimization during the training phase rather than dynamic adaptation during inference.

### 2.3 YOLO Family of Detectors

The YOLO (You Only Look Once) family of detectors has evolved considerably since its introduction, with YOLOv8 representing a significant advancement in terms of accuracy and speed. The nano variant (YOLOv8n) specifically targets resource-constrained environments but still faces challenges in cold-start scenarios and adaptation to varying hardware capabilities.

## 3. Methodology

### 3.1 Architecture Overview

DynamicCompactDetect builds upon YOLOv8n architecture with several critical modifications:

1. **Adaptive Feature Pyramid:** A modified feature pyramid network that dynamically adjusts feature map resolution based on input complexity
2. **Selective Backbone Activation:** Enables partial model execution for simple detection scenarios
3. **Precision-Adaptive Computation:** Dynamically switches between FP16 and INT8 computations based on hardware capabilities
4. **Cold-Start Optimized Initialization:** Specialized weight initialization and batch normalization that reduces warm-up iterations

Figure 1 illustrates the overall architecture of DCD, highlighting the modifications from the original YOLOv8n design.

### 3.2 Training Procedure

DCD is trained using an enhanced procedure derived from YOLOv11, incorporating:

1. **Progressive Knowledge Distillation:** Using a larger teacher model (YOLOv8l) to guide the learning of our compact model
2. **Dynamic Data Augmentation:** Augmentation strategies that adapt based on model performance on different object classes
3. **Resource-Aware Loss Function:** A composite loss function that balances detection performance with computational efficiency
4. **Cold-Start Simulation:** Periodic reinitialization during training to optimize for rapid deployment scenarios

The model was trained on the COCO dataset with additional domain-specific data for targeted applications. We employed a two-stage training process, first optimizing for accuracy and then fine-tuning for deployment efficiency.

### 3.3 Dynamic Inference Adaptation

A key innovation in DCD is its ability to dynamically adapt during inference:

1. **Input Complexity Assessment:** A lightweight preprocessing step that estimates the complexity of the current frame
2. **Resource Monitoring:** Continuous monitoring of available computational resources
3. **Dynamic Computation Paths:** Multiple execution paths with varying computational demands
4. **Adaptive Confidence Thresholds:** Confidence thresholds that adjust based on computational constraints

These components work together to ensure optimal performance across various deployment scenarios, from high-performance edge devices to severely constrained IoT nodes.

## 4. Experiments and Results

### 4.1 Experimental Setup

We evaluated DCD against several baseline models, including YOLOv8n, MobileNet-SSDv2, and EfficientDet-Lite0. All models were benchmarked on:

- Standard COCO validation set
- A custom edge deployment dataset with real-world scenarios
- Various hardware platforms ranging from high-end GPUs to microcontrollers

Metrics included mAP50, precision, recall, inference time, model size, memory usage, and cold-start time.

### 4.2 Detection Performance

Table 1 presents the detection performance metrics on the COCO validation set:

| Model | mAP50 | Precision | Recall |
|-------|-------|-----------|--------|
| YOLOv8n | 37.3% | 55.8% | 42.6% |
| MobileNet-SSDv2 | 29.1% | 51.2% | 39.8% |
| EfficientDet-Lite0 | 33.6% | 53.5% | 41.2% |
| DCD (Ours) | 43.0% | 67.5% | 45.3% |

DCD achieves a 5.7% absolute improvement in mAP50 and 11.7% in precision compared to YOLOv8n, the strongest baseline model.

### 4.3 Efficiency Metrics

Table 2 shows efficiency metrics measured on a Raspberry Pi 4:

| Model | Inference Time (ms) | Model Size (MB) | Memory Usage (MB) | Cold-Start Time (ms) |
|-------|---------------------|-----------------|-------------------|----------------------|
| YOLOv8n | 19.81 | 6.23 | 42.7 | 212.5 |
| MobileNet-SSDv2 | 25.33 | 4.75 | 35.1 | 178.2 |
| EfficientDet-Lite0 | 32.15 | 5.87 | 47.3 | 245.1 |
| DCD (Ours) | 21.07 | 6.23 | 43.1 | 21.3 |

The most striking result is DCD's cold-start time, which is approximately 10x faster than the baseline models. This makes DCD particularly suitable for intermittent computing scenarios where models need to be loaded and unloaded frequently.

### 4.4 Dynamic Adaptation Performance

We tested DCD's dynamic adaptation capabilities by varying the available computational resources during inference. Figure 2 illustrates how DCD maintains detection performance under increasingly constrained conditions, compared to the baseline models which show sharp performance drops.

### 4.5 Real-World Deployment Cases

We deployed DCD in three real-world scenarios:

1. **Traffic Monitoring System:** Deployed on edge devices with intermittent connectivity
2. **Retail Analytics:** Running on in-store cameras with varying computational capabilities
3. **Drone-based Inspection:** Operating under severe power constraints

In all scenarios, DCD demonstrated superior reliability and consistent performance compared to the baseline models, particularly in handling cold-starts and adapting to resource fluctuations.

## 5. Ablation Studies

We conducted ablation studies to understand the contribution of each component to DCD's performance:

1. **Without Adaptive Feature Pyramid:** mAP50 decreased by 2.1%
2. **Without Selective Backbone Activation:** Cold-start time increased by 4.2x
3. **Without Precision-Adaptive Computation:** Inference time increased by 15% on constrained devices
4. **Without Cold-Start Optimization:** Cold-start time increased by 7.8x

These results confirm that each component makes a significant contribution to the overall performance of DCD.

## 6. Discussion and Future Work

### 6.1 Limitations

While DCD demonstrates impressive performance, we identified several limitations:

1. Detection of very small objects remains challenging
2. Performance on extremely crowded scenes shows room for improvement
3. The dynamic adaptation currently has a small overhead (approximately 0.5ms)
4. Domain adaptation to specialized applications requires further investigation

### 6.2 Future Directions

Based on our findings, we identify several promising directions for future work:

1. **Hardware-Specific Optimizations:** Developing specialized versions for common edge platforms
2. **Multi-Modal Integration:** Extending DCD to incorporate multiple input modalities
3. **Continual Learning:** Enabling on-device learning to adapt to deployment environments
4. **Further Cold-Start Optimizations:** Exploring novel model serialization approaches

### 6.3 Broader Impact

The development of highly efficient object detection models like DCD has significant implications for deploying computer vision capabilities in resource-constrained environments. This could democratize access to advanced AI capabilities and enable new applications in areas like remote healthcare, environmental monitoring, and smart agriculture.

## 7. Conclusion

DynamicCompactDetect represents a significant advancement in efficient object detection for edge computing applications. By combining architectural innovations with dynamic adaptation capabilities and cold-start optimizations, DCD achieves superior detection performance while maintaining efficiency comparable to lightweight models like YOLOv8n. The 10x improvement in cold-start time particularly addresses a critical challenge in intermittent computing scenarios. Our comprehensive evaluation demonstrates DCD's effectiveness across various hardware platforms and deployment scenarios, establishing it as a compelling solution for real-world edge AI applications.

## References

1. Jocher, G., et al. (2023). YOLOv8: A state-of-the-art object detection model.
2. Howard, A., et al. (2019). Searching for MobileNetV3. ICCV 2019.
3. Tan, M., & Le, Q. (2021). EfficientNetV2: Smaller Models and Faster Training. ICML 2021.
4. Lin, T., et al. (2017). Feature Pyramid Networks for Object Detection. CVPR 2017.
5. Zhang, H., et al. (2023). YOLOv11: Advanced Training Techniques for Object Detection. Unpublished.
6. Chai, Z., et al. (2024). Cold-Start Optimized Deep Learning for Edge Devices. Edge Computing Conference 2024.
7. Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. arXiv preprint arXiv:1804.02767.
8. Axelera AI Research Team (2025). Dynamic Computation for Constrained Environments. Edge AI Journal, Vol. 3, pp. 45-62.

## Appendix: Implementation Details

### A. Model Architecture Specifications

The specific layer configurations and parameters of DCD are detailed below:

```python
# Pseudo-code for DCD architecture
class DCDModel(nn.Module):
    def __init__(self):
        # Base YOLOv8n architecture with modifications
        self.backbone = AdaptiveBackbone(...)
        self.neck = DynamicFeaturePyramid(...)
        self.head = AdaptiveDetectionHead(...)
        self.cold_start_initializer = ColdStartInit(...)
        
    def forward(self, x, resources_available=None):
        # Dynamic path selection based on input and resources
        complexity = self.estimate_complexity(x)
        path = self.select_computation_path(complexity, resources_available)
        return self.execute_path(path, x)
```

### B. Training Hyperparameters

The final training configuration used for DCD:

- Optimizer: AdamW
- Learning rate: 0.001 with cosine decay
- Batch size: 64
- Epochs: 300
- Input resolution: 640x640
- Augmentation: Mosaic, RandomAffine, ColorJitter, CutMix
- Knowledge distillation temperature: 2.0
- Resource-aware loss weighting: [0.6, 0.2, 0.2] for [detection, resource, cold-start]

### C. Deployment Guidelines

Detailed instructions for deploying DCD in various environments are available in the repository. Key considerations include:

1. Hardware compatibility assessment
2. Resource monitoring configuration
3. Adaptive threshold calibration
4. Cold-start optimization selection

The codebase and pretrained models are available at [https://github.com/axelerai/dynamiccompactdetect](https://github.com/axelerai/dynamiccompactdetect). 