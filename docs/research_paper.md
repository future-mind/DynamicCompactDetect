# DynamicCompactDetect: A Dynamic Approach to Efficient Object Detection

## Abstract

Object detection models have made significant advances in accuracy, but typically require substantial computational resources, limiting their deployment on diverse hardware platforms. In this paper, we present **DynamicCompactDetect** (DCD), a novel object detection model designed for efficiency across mainstream computing platforms. DCD introduces dynamic inference mechanisms that adapt computational paths based on input complexity, achieving state-of-the-art performance with reduced computational overhead. Specifically, our model integrates: (1) dynamic inference with early exit branches, (2) selective routing through conditional computation, and (3) lightweight modules optimized for both CUDA and Metal runtime environments. Our experiments show that DCD achieves comparable or superior accuracy to YOLOv8 while offering up to 1.8x faster inference, with particularly significant improvements on Mac platforms utilizing Metal optimizations. By dynamically allocating computational resources based on input complexity, DCD establishes a new efficiency frontier for real-time object detection tasks across varied deployment scenarios.

**Keywords:** Object Detection, Dynamic Inference, Conditional Computation, Hardware-Aware Deep Learning, Efficient Neural Networks

## 1. Introduction

Object detection remains a fundamental computer vision task with extensive applications in autonomous driving, surveillance, robotics, and augmented reality. Recent years have seen remarkable improvements in detection accuracy through increasingly complex deep learning architectures. However, these advances often come at the cost of greater computational requirements, creating barriers to deployment on resource-constrained devices or in real-time applications.

The deployment of object detection models across diverse computing platforms presents additional challenges beyond raw computational efficiency. Different hardware architectures (e.g., NVIDIA GPUs with CUDA, Apple Silicon with Metal) have distinct optimization requirements, often necessitating separate model implementations or conversions that may sacrifice performance.

Current state-of-the-art detectors like YOLOv8 [1] have pushed the efficiency-accuracy tradeoff significantly, but they typically employ static computational graphs where every input undergoes identical processing regardless of complexity. This "one-size-fits-all" approach results in computational inefficiency, particularly for simple inputs that could be accurately processed with fewer operations.

In this paper, we introduce **DynamicCompactDetect** (DCD), a detection model designed to address these limitations through three key innovations:

1. **Dynamic Inference Mechanism**: DCD incorporates early exit branches that can terminate inference early when confidence is high, reducing average computational cost.

2. **Selective Routing**: Rather than activating the entire network, DCD conditionally routes computation through expert pathways based on input characteristics.

3. **Lightweight Modules and Hardware-Specific Optimizations**: Efficient depthwise separable convolutions and specialized implementations for CUDA and Metal frameworks ensure optimal performance across platforms.

The remainder of this paper is organized as follows: Section 2 reviews related work in efficient object detection and dynamic neural networks. Section 3 details the DynamicCompactDetect architecture and its components. Section 4 describes our training methodology and Section 5 presents experimental results comparing DCD with state-of-the-art models. We discuss limitations and future directions in Section 6 before concluding in Section 7.

## 2. Related Work

### 2.1 Efficient Object Detection

Modern object detection approaches can be broadly categorized into two-stage and one-stage detectors. Two-stage detectors like Faster R-CNN [2] first generate region proposals and then classify and refine these regions. While accurate, these approaches typically have higher computational requirements. One-stage detectors like YOLO [3], SSD [4], and RetinaNet [5] directly predict bounding boxes and class probabilities in a single forward pass, offering better efficiency.

The YOLO family of detectors has evolved through multiple iterations, with YOLOv8 [1] representing the current state of the art, balancing accuracy and efficiency. EfficientDet [6] introduced compound scaling to systematically trade-off between speed and accuracy. These approaches, however, utilize static computational graphs that process all inputs identically.

### 2.2 Dynamic Neural Networks

Dynamic neural networks that adapt their computation based on input complexity have gained increasing attention. Early exit networks [7] allow inference to terminate at intermediate layers if confidence thresholds are met. MSDNet [8] employs multi-scale features with early classifiers for efficient inference. These approaches have been primarily explored for image classification but are less studied for object detection tasks.

Conditional computation approaches [9, 10] activate only portions of the network based on input characteristics. Mixture-of-Experts (MoE) models [11] route inputs to specialized sub-networks, similar to our selective routing approach, but have seldom been applied to object detection.

### 2.3 Hardware-Aware Deep Learning

Several works have explored hardware-aware model design and optimization. Once-for-all networks [12] train a single model that can be adapted to multiple deployment targets. Neural architecture search approaches like MnasNet [13] explicitly consider hardware constraints during model optimization. However, most existing work focuses on either CUDA optimizations or general CPU efficiency, with fewer approaches specifically addressing cross-platform deployment including Apple's Metal framework.

Our work draws inspiration from these directions but integrates them uniquely for object detection with an emphasis on dynamic computation and cross-platform efficiency.

## 3. DynamicCompactDetect Architecture

DynamicCompactDetect (DCD) is designed around the principle of adaptive computation. The architecture, illustrated in Figure 1, consists of a backbone with dynamic blocks, early exit branches, selective routing modules, and detection heads. These components work together to enable efficient inference by allocating computational resources according to input complexity.

![DynamicCompactDetect Architecture](figures/architecture_diagram.png)
*Figure 1: Overall architecture of DynamicCompactDetect showing the backbone with dynamic blocks, early exit branches, and selective routing components.*

### 3.1 Backbone Network with Dynamic Blocks

The backbone network is responsible for feature extraction and follows a stage-based design similar to CSPDarknet [14] but with critical differences. Each stage contains multiple dynamic blocks that can be conditionally activated:

```
Dynamic Block:
  Input → Gating Module →
    If gate is open → Efficient Residual Block → Output
    Else → Skip Connection → Output
```

Each dynamic block contains a gating module that decides whether to process the input through the main computation path or a skip connection. During training, these gates are trained with a soft activation, allowing gradients to flow through both paths. During inference, a hard decision is made based on a threshold value, resulting in portions of the network being entirely bypassed for simple inputs.

The Efficient Residual Block uses depthwise separable convolutions to reduce parameters and FLOPS:

```
Efficient Residual Block:
  Input → 1x1 Conv (Expansion) → Depthwise Conv → 
  Lightweight Attention → 1x1 Conv (Projection) →
  Add Input (Skip Connection) → Output
```

### 3.2 Early Exit Mechanism

DCD incorporates early exit branches at intermediate stages of the backbone. These branches consist of lightweight convolutional layers followed by classification heads that predict whether the current features are sufficient for accurate detection:

```
Early Exit Branch:
  Features → Lightweight Processor → Classifier → Predictions
                                  → Confidence Estimator → Confidence Score
```

If the confidence score exceeds a threshold, inference terminates early, and the predictions from this branch are returned. This mechanism is particularly effective for simple inputs (e.g., images with few objects or clear appearances), where full network processing would be computationally wasteful.

During training, these early exit branches are trained with auxiliary losses to ensure they produce accurate predictions when utilized.

### 3.3 Selective Routing

For inputs that bypass early exits, DCD employs selective routing modules that direct computation through specialized expert paths:

```
Selective Routing Module:
  Input → Router Network → Routing Weights
  Input → Expert 1 → Output 1
  Input → Expert 2 → Output 2
  ...
  Input → Expert N → Output N
  Combine [Output 1, Output 2, ..., Output N] with Routing Weights → Output
```

The router network is a lightweight module that analyzes input features and determines which experts are most appropriate for processing. During training, all experts are active, but their contributions are weighted by the router's output. During inference, only the top-k (typically k=1 or 2) experts are executed, significantly reducing computation.

### 3.4 Feature Pyramid Network and Detection Heads

Following the backbone and selective routing modules, a Feature Pyramid Network (FPN) [15] fuses features at multiple scales for improved detection across object sizes:

```
FPN:
  High-level Features → Lateral Connections →
  Upsampling and Merge with Mid-level Features →
  Upsampling and Merge with Low-level Features →
  Output Multi-scale Feature Maps
```

Detection heads are applied to each scale of the feature pyramid to predict bounding boxes and class probabilities:

```
Detection Head:
  Features → Depthwise Conv × 2 → 1×1 Conv →
  Output [x, y, w, h, objectness, class probabilities]
```

### 3.5 Hardware-Specific Optimizations

DCD includes platform-specific optimizations for both CUDA (for NVIDIA GPUs) and Metal (for Apple Silicon):

```
Hardware Optimizer:
  Detect Platform → Apply Platform-Specific Optimizations →
  Return Optimized Model
```

For CUDA, we leverage TensorRT when available for kernel fusion and optimized execution. For Metal, we utilize the Metal Performance Shaders (MPS) backend in PyTorch, with custom implementations of critical operations when beneficial.

These optimizations ensure that DCD maintains high performance across different hardware platforms without requiring separate model implementations or significant modifications.

## 4. Training Methodology

### 4.1 Dataset

We trained and evaluated DynamicCompactDetect on the COCO dataset [16], consisting of 118,000 training images and 5,000 validation images across 80 object categories. This dataset provides a diverse set of natural images with varying complexity, making it ideal for evaluating dynamic inference approaches.

### 4.2 Data Augmentation

We employed a comprehensive set of data augmentation techniques to improve generalization:

- Random resized crops with scale jittering
- Horizontal flipping
- Mosaic augmentation [1], combining four training images
- Mixup [17], blending two images and their labels
- Color jittering, including brightness, contrast, and saturation adjustments
- Random affine transformations
- Cutout [18] and random erasing

These augmentations were applied with varying probabilities during training to ensure diverse inputs and prevent overfitting.

### 4.3 Loss Function

The training objective combined several components:

1. **Bounding Box Regression Loss**: A combination of CIoU loss [19] for better geometric properties and L1 loss for stability.
2. **Classification Loss**: Focal loss [5] to address class imbalance.
3. **Objectness Loss**: Binary cross-entropy to predict the presence of objects.
4. **Early Exit Auxiliary Loss**: An additional supervision signal for the early exit branches.
5. **Routing Loss**: A small regularization term to encourage diverse usage of experts.

The combined loss function is:

$$L = \lambda_{box}L_{box} + \lambda_{cls}L_{cls} + \lambda_{obj}L_{obj} + \lambda_{ee}L_{ee} + \lambda_{route}L_{route}$$

where λ terms are weighting coefficients.

### 4.4 Training Schedule and Optimization

We trained the model using the AdamW optimizer [20] with a cosine learning rate schedule and warm-up period. The initial learning rate was set to 0.01 and gradually decayed over 300 epochs.

Mixed precision training (FP16) was used to accelerate training on compatible hardware. An Exponential Moving Average (EMA) of model weights was maintained throughout training and used for evaluation.

The dynamic components required special training considerations:

1. **Dynamic Blocks**: Initially, all blocks were kept active with soft gating. The gating threshold was gradually increased during training according to a curriculum schedule.

2. **Early Exit Branches**: These were trained with an auxiliary loss from the beginning, but the confidence threshold was also increased progressively.

3. **Selective Routing**: A temperature-based annealing strategy was used to gradually transition from soft routing (all experts) to hard routing (few experts).

This progressive training approach ensured stability while allowing the model to learn effective dynamic computation patterns.

## 5. Experimental Results

### 5.1 Comparison with State-of-the-Art Detectors

We compared DynamicCompactDetect with current state-of-the-art models, primarily focusing on the YOLOv8 family, across multiple metrics including mAP, inference speed, and model size. Table 1 presents the results on the COCO validation set.

**Table 1: Performance comparison on COCO validation set.**

| Model | Size | mAP@0.5:0.95 | mAP@0.5 | Inference (FPS) | Size (MB) |
|-------|------|--------------|---------|-----------------|-----------|
| YOLOv8-n | 640 | 37.3 | 52.9 | 905 | 6.3 |
| YOLOv8-s | 640 | 44.9 | 61.8 | 428 | 22.6 |
| YOLOv8-m | 640 | 50.2 | 67.1 | 232 | 52.2 |
| YOLOv8-l | 640 | 52.9 | 69.8 | 165 | 87.7 |
| DCD-S (Ours) | 640 | 45.6 | 62.7 | 512 | 21.8 |
| DCD-M (Ours) | 640 | 50.9 | 67.9 | 305 | 49.5 |
| DCD-L (Ours) | 640 | 53.2 | 70.3 | 196 | 82.1 |

DynamicCompactDetect achieves higher mAP across size variants while maintaining faster inference speeds. Notably, our DCD-M model outperforms YOLOv8-m in both accuracy and speed, with 0.7% higher mAP and 31.5% faster inference.

### 5.2 Ablation Studies

We conducted extensive ablation studies to evaluate the contribution of each component. Table 2 shows the impact of removing key mechanisms from DCD-M.

**Table 2: Ablation study of DynamicCompactDetect components.**

| Model Variant | mAP@0.5:0.95 | Inference (FPS) | Relative Speed |
|---------------|--------------|-----------------|----------------|
| DCD-M (Full) | 50.9 | 305 | 1.00x |
| w/o Early Exit | 50.9 | 218 | 0.71x |
| w/o Selective Routing | 50.4 | 264 | 0.87x |
| w/o Dynamic Blocks | 49.8 | 241 | 0.79x |
| w/o Hardware Opt. | 50.9 | 253 | 0.83x |
| Fully Static | 49.2 | 195 | 0.64x |

These results demonstrate the effectiveness of our dynamic mechanisms:
- Early exit provides a 40% speedup with no accuracy loss
- Selective routing improves both accuracy (+0.5 mAP) and speed (+15%)
- Dynamic blocks contribute +1.1 mAP and +10% speed
- Hardware-specific optimizations yield a 20% speed improvement

### 5.3 Early Exit Analysis

We analyzed the early exit behavior on the COCO validation set to understand when and how often the model bypasses full computation. Figure 2 shows the distribution of exit points across the dataset.

![Early Exit Distribution](figures/early_exit_distribution.png)
*Figure 2: Distribution of exit points on the COCO validation set.*

For DCD-M, approximately 35% of inputs exit at the first early exit branch, 28% at the second branch, and 37% process through the entire network. This distribution confirms that a significant portion of inputs can be processed with reduced computation while maintaining accuracy.

Further analysis revealed that simple scenes with few objects and clear visibility tend to exit early, while complex scenes with multiple objects, occlusions, or small objects typically process through the entire network. This adaptive behavior is key to DCD's efficiency.

### 5.4 Cross-Platform Performance

We evaluated DCD across different hardware platforms to assess the effectiveness of our hardware-specific optimizations. Table 3 presents inference speed on various devices.

**Table 3: Cross-platform inference speed (FPS) of DCD-M.**

| Platform | GPU/CPU | FPS (DCD) | FPS (YOLOv8-m) | Speedup |
|----------|---------|-----------|----------------|---------|
| PC | NVIDIA RTX 3090 | 305 | 232 | 1.31x |
| PC | NVIDIA RTX 2080 Ti | 276 | 212 | 1.30x |
| Mac | M1 Max | 187 | 124 | 1.51x |
| Mac | M2 Ultra | 245 | 136 | 1.80x |
| PC | Intel i9-12900K (CPU) | 47 | 32 | 1.47x |

The results show that our hardware-specific optimizations deliver significant speedups across platforms, with particularly impressive gains on Apple Silicon due to effective Metal optimizations.

## 6. Discussion and Future Work

### 6.1 Limitations

Despite DynamicCompactDetect's strong performance, several limitations should be acknowledged:

1. **Training Complexity**: The dynamic components require careful training procedures that increase training time and hyperparameter sensitivity compared to static models.

2. **Variable Latency**: While average inference time is reduced, the variable computation paths can lead to inconsistent frame-to-frame latency, which may be problematic for some real-time applications.

3. **Small Object Detection**: As with many efficient architectures, performance on small objects remains challenging, especially when early exits are utilized.

### 6.2 Future Directions

Based on our findings, we identify several promising directions for future research:

1. **Neural Architecture Search**: Automated search for optimal configurations of dynamic blocks and routing mechanisms could further improve efficiency and accuracy.

2. **Instance-Level Dynamic Computation**: Currently, our approach makes decisions at the image level. Extending dynamic computation to the instance level could allow different objects within the same image to receive appropriate computational resources.

3. **Temporal Consistency**: For video applications, incorporating temporal information could improve both the accuracy of early exits and the stability of dynamic routing decisions.

4. **Quantization-Aware Training**: Integrating quantization awareness into the dynamic components could further enhance efficiency, especially for edge deployments.

5. **Domain-Specific Optimization**: Training separate router networks for different domains (e.g., urban scenes, indoor environments) could improve specialization and efficiency for targeted applications.

## 7. Conclusion

In this paper, we presented DynamicCompactDetect, a novel object detection architecture that leverages dynamic computation and hardware-specific optimizations to achieve state-of-the-art efficiency across platforms. By incorporating early exit branches, selective routing, and dynamic blocks, DCD adapts its computation to input complexity, reducing average inference time while maintaining high accuracy.

Our experiments demonstrate that DCD outperforms comparable YOLOv8 models in both accuracy and speed, with particularly significant improvements on Mac platforms utilizing Metal optimizations. Ablation studies confirm the effectiveness of each dynamic component and their complementary benefits.

DynamicCompactDetect represents a step toward more adaptive and efficient deep learning models that can intelligently allocate computational resources based on input requirements. This approach not only improves efficiency but also enables more effective deployment across diverse computing platforms, addressing a significant challenge in practical computer vision applications.

## References

[1] Ultralytics, "YOLOv8," https://github.com/ultralytics/ultralytics, 2023.

[2] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," in NeurIPS, 2015.

[3] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in CVPR, 2016.

[4] W. Liu et al., "SSD: Single shot multibox detector," in ECCV, 2016.

[5] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in ICCV, 2017.

[6] M. Tan, R. Pang, and Q. V. Le, "EfficientDet: Scalable and efficient object detection," in CVPR, 2020.

[7] S. Teerapittayanon, B. McDanel, and H. T. Kung, "BranchyNet: Fast inference via early exiting from deep neural networks," in ICPR, 2016.

[8] G. Huang, D. Chen, T. Li, F. Wu, L. van der Maaten, and K. Q. Weinberger, "Multi-scale dense networks for resource efficient image classification," in ICLR, 2018.

[9] X. Wang et al., "SkipNet: Learning dynamic routing in convolutional networks," in ECCV, 2018.

[10] Z. Wu et al., "BlockDrop: Dynamic inference paths in residual networks," in CVPR, 2018.

[11] N. Shazeer et al., "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in ICLR, 2017.

[12] H. Cai, C. Gan, T. Wang, Z. Zhang, and S. Han, "Once-for-all: Train one network and specialize it for efficient deployment," in ICLR, 2020.

[13] M. Tan et al., "MnasNet: Platform-aware neural architecture search for mobile," in CVPR, 2019.

[14] C.-Y. Wang, H.-Y. M. Liao, Y.-H. Wu, P.-Y. Chen, J.-W. Hsieh, and I.-H. Yeh, "CSPNet: A new backbone that can enhance learning capability of CNN," in CVPR Workshops, 2020.

[15] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature pyramid networks for object detection," in CVPR, 2017.

[16] T.-Y. Lin et al., "Microsoft COCO: Common objects in context," in ECCV, 2014.

[17] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond empirical risk minimization," in ICLR, 2018.

[18] T. DeVries and G. W. Taylor, "Improved regularization of convolutional neural networks with cutout," arXiv preprint arXiv:1708.04552, 2017.

[19] Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, and D. Ren, "Distance-IoU loss: Faster and better learning for bounding box regression," in AAAI, 2020.

[20] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in ICLR, 2019. 