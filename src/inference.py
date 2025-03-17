import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional
import time
import json

from src.model import create_yolov10_model
from src.utils import non_max_suppression, scale_boxes, xywh2xyxy, visualize_detection, time_sync

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class YOLOv10Predictor:
    """
    Class for making predictions with YOLOv10-style model.
    """
    def __init__(
        self,
        model_path: str,
        num_classes: int = 80,
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        device: str = None,
        class_names: List[str] = None,
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to model weights
            num_classes: Number of classes
            img_size: Image size for inference
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections
            device: Device to run inference on
            class_names: List of class names
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model_path = model_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        
        # Load model configuration and weights
        if model_path.endswith('model_deploy.pt'):
            # Load a deployment-optimized model
            self.model = self.load_optimized_model(model_path)
        else:
            # Load weights only (assuming model structure is defined)
            self.model = self.load_model(model_path, num_classes)
        
        # Set class names
        self.class_names = class_names if class_names else COCO_CLASSES
    
    def load_model(self, model_path: str, num_classes: int) -> torch.nn.Module:
        """
        Load model from weights file.
        
        Args:
            model_path: Path to model weights
            num_classes: Number of classes
            
        Returns:
            Loaded model
        """
        # Load checkpoint
        ckpt = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in ckpt:
            # Get model parameters from checkpoint if available
            if 'config' in ckpt:
                config = ckpt['config']
                width_multiple = config.get('width_multiple', 0.25)
                depth_multiple = config.get('depth_multiple', 0.33)
                activation = config.get('activation', 'silu')
            else:
                # Default to nano variant if not specified
                width_multiple = 0.25
                depth_multiple = 0.33
                activation = 'silu'
            
            # Create model with appropriate parameters
            model = create_yolov10_model(
                num_classes=num_classes,
                width_multiple=width_multiple,
                depth_multiple=depth_multiple,
                activation=activation
            )
            
            # Load weights
            state_dict = ckpt['model_state_dict']
            model.load_state_dict(state_dict)
        else:
            # Assume the checkpoint is the model itself
            model = ckpt
        
        # Set model to evaluation mode
        model.eval()
        model.to(self.device)
        
        # Attempt to fuse layers for optimized inference if not already done
        if hasattr(model, 'fuse') and not getattr(model, '_is_fused', False):
            print("Fusing layers for optimized inference...")
            model.fuse()
            model._is_fused = True
        
        return model
    
    def load_optimized_model(self, model_path: str) -> torch.nn.Module:
        """
        Load a deployment-optimized model.
        
        Args:
            model_path: Path to optimized model
            
        Returns:
            Loaded model
        """
        # Load the optimized model
        ckpt = torch.load(model_path, map_location=self.device)
        
        # Get configuration
        config = ckpt['config']
        num_classes = config.get('num_classes', 80)
        width_multiple = config.get('width_multiple', 0.25)
        depth_multiple = config.get('depth_multiple', 0.33)
        activation = config.get('activation', 'silu')
        
        # Create model
        model = create_yolov10_model(
            num_classes=num_classes,
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            activation=activation
        )
        
        # Load weights
        model.load_state_dict(ckpt['model_state_dict'])
        
        # Set model to evaluation mode
        model.eval()
        model.to(self.device)
        
        return model
    
    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image tensor, original image size, and resized image size
        """
        # Original image size
        orig_h, orig_w = image.shape[:2]
        
        # Resize and pad image
        img = letterbox(image, new_shape=self.img_size)[0]
        
        # Convert to RGB and normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        # Add batch dimension
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        img = img.to(self.device)
        
        return img, (orig_h, orig_w), img.shape[2:]
    
    @torch.no_grad()
    def predict(
        self, 
        image: np.ndarray, 
        augment: bool = False,
        classes: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Run prediction on image.
        
        Args:
            image: Input image
            augment: Whether to use augmented inference
            classes: Filter predictions by class
            
        Returns:
            List of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        # Preprocess image
        img, orig_size, resized_size = self.preprocess(image)
        
        # Time inference
        t1 = time_sync()
        
        # Inference
        pred = self.model(img, augment=augment)
        
        # Apply NMS
        pred = non_max_suppression(
            pred, 
            self.conf_thres, 
            self.iou_thres,
            classes=classes,
            max_det=self.max_det
        )
        
        t2 = time_sync()
        
        # Print time (inference-only)
        if len(pred[0]):
            print(f"Inference time: {(t2 - t1) * 1000:.1f}ms ({1 / (t2 - t1):.1f} FPS)")
        
        # Process detections
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(resized_size, det[:, :4], orig_size)
        
        return pred
    
    def predict_batch(
        self, 
        images: List[np.ndarray], 
        augment: bool = False,
        classes: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Run prediction on a batch of images.
        
        Args:
            images: List of input images
            augment: Whether to use augmented inference
            classes: Filter predictions by class
            
        Returns:
            List of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        batch_results = []
        for image in images:
            results = self.predict(image, augment, classes)
            batch_results.extend(results)
        return batch_results
    
    def visualize(
        self,
        image: np.ndarray, 
        detections: torch.Tensor, 
        line_thickness: int = 2,
        font_scale: float = 0.5,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detection results on image.
        
        Args:
            image: Original image
            detections: Detection results
            line_thickness: Line thickness
            font_scale: Font scale
            save_path: Path to save visualization (optional)
            
        Returns:
            Image with visualized detections
        """
        if len(detections):
            # Get boxes, labels, and scores
            boxes = detections[:, :4].cpu()
            scores = detections[:, 4].cpu()
            labels = detections[:, 5].cpu()
            
            # Visualize
            result_img = visualize_detection(
                image.copy(), 
                boxes, 
                labels, 
                scores, 
                self.class_names,
                line_thickness=line_thickness,
                font_scale=font_scale
            )
            
            # Save if path is provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, result_img)
                print(f"Result saved to {save_path}")
            
            return result_img
        else:
            print("No detections found")
            
            # Save original image if path is provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, image)
                print(f"Original image saved to {save_path} (no detections)")
            
            return image
    
    def benchmark(self, image: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            image: Input image
            num_runs: Number of runs for benchmarking
            
        Returns:
            Dictionary of benchmark results
        """
        # Warmup
        for _ in range(10):
            _ = self.predict(image)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            t1 = time_sync()
            _ = self.predict(image)
            t2 = time_sync()
            times.append((t2 - t1) * 1000)  # ms
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        return {
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'fps': 1000 / avg_time,
            'num_runs': num_runs
        }


def letterbox(
    img, 
    new_shape=(640, 640), 
    color=(114, 114, 114), 
    auto=True, 
    scaleFill=False, 
    scaleup=True, 
    stride=32
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize and pad image while maintaining aspect ratio.
    
    Args:
        img: Image to resize
        new_shape: Target size
        color: Color for padding
        auto: Auto compute new shape based on stride
        scaleFill: Scale to fill new shape (disables aspect ratio preservation)
        scaleup: Allow scaling up (only scales down if False)
        stride: Stride for auto computation
        
    Returns:
        Resized and padded image, ratio, padding
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def main():
    """
    Run inference on image or directory.
    """
    parser = argparse.ArgumentParser(description='YOLOv10 Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--image-path', type=str, required=True, help='Path to image or directory of images')
    parser.add_argument('--output-path', type=str, default='outputs/inference', help='Path to save output')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device to use (cuda or cpu)')
    parser.add_argument('--classes', type=int, nargs='+', default=None, help='Filter by class')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--save-json', action='store_true', help='Save detections as JSON')
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = YOLOv10Predictor(
        model_path=args.model_path,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
    )
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Check if image path is a file or directory
    if os.path.isfile(args.image_path):
        # Single image inference
        image = cv2.imread(args.image_path)
        if image is None:
            print(f"Error: Could not read image {args.image_path}")
            return
        
        # Run benchmark if requested
        if args.benchmark:
            benchmark_results = predictor.benchmark(image)
            print(f"Benchmark results:")
            for k, v in benchmark_results.items():
                print(f"  {k}: {v:.2f}")
            
            # Save benchmark results
            benchmark_path = os.path.join(args.output_path, 'benchmark.json')
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark_results, f, indent=4)
            print(f"Benchmark results saved to {benchmark_path}")
        
        # Run prediction
        detections = predictor.predict(image, classes=args.classes)[0]
        
        # Visualize and save
        file_name = os.path.basename(args.image_path)
        save_path = os.path.join(args.output_path, file_name)
        result_img = predictor.visualize(image, detections, save_path=save_path)
        
        # Save detections as JSON if requested
        if args.save_json:
            detections_json = []
            for det in detections:
                box = det[:4].tolist()
                conf = det[4].item()
                cls_id = int(det[5].item())
                cls_name = predictor.class_names[cls_id]
                
                detections_json.append({
                    'bbox': box,
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': cls_name
                })
            
            json_path = os.path.join(args.output_path, os.path.splitext(file_name)[0] + '.json')
            with open(json_path, 'w') as f:
                json.dump(detections_json, f, indent=4)
            print(f"Detections saved to {json_path}")
    
    else:
        # Directory of images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_paths = [
            os.path.join(args.image_path, f) 
            for f in os.listdir(args.image_path) 
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        if not image_paths:
            print(f"Error: No images found in {args.image_path}")
            return
        
        print(f"Processing {len(image_paths)} images...")
        
        for img_path in image_paths:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Could not read image {img_path}")
                continue
            
            # Run prediction
            detections = predictor.predict(image, classes=args.classes)[0]
            
            # Visualize and save
            file_name = os.path.basename(img_path)
            save_path = os.path.join(args.output_path, file_name)
            result_img = predictor.visualize(image, detections, save_path=save_path)
            
            # Save detections as JSON if requested
            if args.save_json:
                detections_json = []
                for det in detections:
                    box = det[:4].tolist()
                    conf = det[4].item()
                    cls_id = int(det[5].item())
                    cls_name = predictor.class_names[cls_id]
                    
                    detections_json.append({
                        'bbox': box,
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': cls_name
                    })
                
                json_path = os.path.join(args.output_path, os.path.splitext(file_name)[0] + '.json')
                with open(json_path, 'w') as f:
                    json.dump(detections_json, f, indent=4)
        
        print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main() 