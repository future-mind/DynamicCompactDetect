#!/usr/bin/env python3
"""
Tiled Detector for Tiny Object Detection

This script implements a tiled detection approach to improve tiny object detection performance:
1. Divides large images into smaller tiles with overlap
2. Processes each tile in parallel with DynamicCompactDetect
3. Merges detection results and handles duplicate detections
4. Compares performance with regular (non-tiled) detection

Recommended datasets:
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
- TinyPerson: https://github.com/ucas-vg/TinyBenchmark
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
import torch.multiprocessing as mp

# Define paths
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'
DATA_DIR = ROOT_DIR / 'data'
TEST_IMAGES_DIR = DATA_DIR / 'test_images'

# Colors for visualization
COLORS = {
    'regular': (0, 0, 255),   # Red
    'tiled': (0, 255, 0),     # Green
    'overlap': (255, 0, 255)  # Magenta
}

class TiledDetector:
    """Implements tiled detection for improved tiny object detection."""
    
    def __init__(
        self, 
        model_path, 
        tile_size=640, 
        overlap=0.2, 
        conf_threshold=0.25, 
        iou_threshold=0.45
    ):
        """
        Initialize the TiledDetector.
        
        Args:
            model_path: Path to the model weights file
            tile_size: Size of each square tile (default: 640)
            overlap: Overlap between adjacent tiles as a fraction (default: 0.2)
            conf_threshold: Confidence threshold for detections (default: 0.25)
            iou_threshold: IoU threshold for non-maximum suppression (default: 0.45)
        """
        self.model_path = model_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the detection model."""
        try:
            model = YOLO(self.model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def _generate_tiles(self, image):
        """
        Generate overlapping tiles from the input image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of (tile, (x1, y1, x2, y2)) tuples, where (x1, y1, x2, y2) are
            the coordinates of the tile in the original image
        """
        height, width = image.shape[:2]
        stride = int(self.tile_size * (1 - self.overlap))
        
        tiles = []
        coords = []
        
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                x1, y1 = x, y
                x2, y2 = min(x + self.tile_size, width), min(y + self.tile_size, height)
                
                # Skip tiles that are too small
                if x2 - x1 < self.tile_size // 2 or y2 - y1 < self.tile_size // 2:
                    continue
                
                tile = image[y1:y2, x1:x2]
                tiles.append((tile, (x1, y1, x2, y2)))
                coords.append((x1, y1, x2, y2))
        
        return tiles, coords
    
    def _process_tile(self, tile_data):
        """
        Process a single tile and return detections.
        
        Args:
            tile_data: Tuple of (tile, (x1, y1, x2, y2))
            
        Returns:
            List of detections, each as (x1, y1, x2, y2, conf, cls)
            adjusted to the original image coordinates
        """
        tile, (tile_x1, tile_y1, tile_x2, tile_y2) = tile_data
        
        # Run inference on the tile
        results = self.model(tile, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # Extract and adjust detections
        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
                
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = box.conf[0].cpu().numpy().item()
                cls = box.cls[0].cpu().numpy().item()
                
                # Adjust coordinates to the original image
                x1 += tile_x1
                y1 += tile_y1
                x2 += tile_x1
                y2 += tile_y1
                
                detections.append((x1, y1, x2, y2, conf, cls))
        
        return detections
    
    def _merge_detections(self, all_detections):
        """
        Merge detections from all tiles and apply NMS.
        
        Args:
            all_detections: List of detection lists from each tile
            
        Returns:
            List of final detections after merging and NMS
        """
        # Flatten the list of lists
        detections = [det for sublist in all_detections for det in sublist]
        
        if not detections:
            return []
        
        # Prepare detections for NMS
        boxes = np.array([det[:4] for det in detections])
        scores = np.array([det[4] for det in detections])
        classes = np.array([det[5] for det in detections])
        
        # Convert to PyTorch tensors
        boxes_tensor = torch.from_numpy(boxes).float()
        scores_tensor = torch.from_numpy(scores).float()
        classes_tensor = torch.from_numpy(classes).float()
        
        # Perform class-wise NMS
        unique_classes = classes_tensor.unique()
        final_detections = []
        
        for cls in unique_classes:
            cls_mask = (classes_tensor == cls)
            cls_boxes = boxes_tensor[cls_mask]
            cls_scores = scores_tensor[cls_mask]
            
            # Get the indices of the boxes to keep after NMS
            keep_indices = torchvision_nms(cls_boxes, cls_scores, self.iou_threshold)
            
            # Extract the kept detections
            for idx in keep_indices:
                idx = idx.item()
                box = cls_boxes[idx].tolist()
                score = cls_scores[idx].item()
                final_detections.append((*box, score, cls.item()))
        
        return final_detections
    
    def detect_regular(self, image):
        """
        Perform regular (non-tiled) detection on the image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of detections, each as (x1, y1, x2, y2, conf, cls)
        """
        start_time = time.time()
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
        end_time = time.time()
        
        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
                
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = box.conf[0].cpu().numpy().item()
                cls = box.cls[0].cpu().numpy().item()
                
                detections.append((x1, y1, x2, y2, conf, cls))
        
        return detections, end_time - start_time
    
    def detect_tiled(self, image):
        """
        Perform tiled detection on the image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of detections, each as (x1, y1, x2, y2, conf, cls)
        """
        start_time = time.time()
        
        # Generate tiles
        tiles, coords = self._generate_tiles(image)
        
        # Process tiles in parallel
        all_detections = []
        with ThreadPoolExecutor() as executor:
            future_to_tile = {executor.submit(self._process_tile, tile_data): i for i, tile_data in enumerate(tiles)}
            for future in as_completed(future_to_tile):
                tile_idx = future_to_tile[future]
                try:
                    detections = future.result()
                    all_detections.append(detections)
                except Exception as e:
                    print(f"Error processing tile {tile_idx}: {e}")
        
        # Merge detections
        final_detections = self._merge_detections(all_detections)
        
        end_time = time.time()
        return final_detections, end_time - start_time, coords
    
    def visualize_comparison(self, image, regular_detections, tiled_detections, coords=None, output_path=None):
        """
        Visualize and compare regular and tiled detections.
        
        Args:
            image: Input image (numpy array)
            regular_detections: List of regular detections
            tiled_detections: List of tiled detections
            coords: List of tile coordinates for visualization
            output_path: Path to save the output image (optional)
        
        Returns:
            Visualization image
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Draw tile boundaries if coords are provided
        if coords:
            for x1, y1, x2, y2 in coords:
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), COLORS['overlap'], 1)
        
        # Draw regular detections
        for x1, y1, x2, y2, conf, cls in regular_detections:
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), COLORS['regular'], 2)
            label = f"{int(cls)}: {conf:.2f}"
            cv2.putText(vis_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['regular'], 2)
        
        # Draw tiled detections
        for x1, y1, x2, y2, conf, cls in tiled_detections:
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), COLORS['tiled'], 2)
            label = f"{int(cls)}: {conf:.2f}"
            cv2.putText(vis_image, label, (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['tiled'], 2)
        
        # Add legend
        legend_y = 30
        cv2.putText(vis_image, "Regular", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['regular'], 2)
        cv2.putText(vis_image, "Tiled", (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['tiled'], 2)
        cv2.putText(vis_image, "Tile Boundary", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['overlap'], 2)
        
        # Save the output image if a path is provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image
    
    def analyze_results(self, regular_detections, tiled_detections, regular_time, tiled_time):
        """
        Analyze and compare detection results.
        
        Args:
            regular_detections: List of regular detections
            tiled_detections: List of tiled detections
            regular_time: Time taken for regular detection
            tiled_time: Time taken for tiled detection
            
        Returns:
            Dictionary with analysis results
        """
        # Count detections
        regular_count = len(regular_detections)
        tiled_count = len(tiled_detections)
        
        # Compute average confidence
        regular_conf = np.mean([det[4] for det in regular_detections]) if regular_detections else 0
        tiled_conf = np.mean([det[4] for det in tiled_detections]) if tiled_detections else 0
        
        # Analyze box sizes
        regular_sizes = [abs((det[2] - det[0]) * (det[3] - det[1])) for det in regular_detections] if regular_detections else []
        tiled_sizes = [abs((det[2] - det[0]) * (det[3] - det[1])) for det in tiled_detections] if tiled_detections else []
        
        regular_avg_size = np.mean(regular_sizes) if regular_sizes else 0
        tiled_avg_size = np.mean(tiled_sizes) if tiled_sizes else 0
        
        # Count tiny objects (area < 32*32 pixels)
        tiny_threshold = 32 * 32
        regular_tiny_count = sum(1 for size in regular_sizes if size < tiny_threshold)
        tiled_tiny_count = sum(1 for size in tiled_sizes if size < tiny_threshold)
        
        # Create analysis results
        analysis = {
            'detection_counts': {
                'regular': regular_count,
                'tiled': tiled_count,
                'difference': tiled_count - regular_count
            },
            'confidence': {
                'regular': regular_conf,
                'tiled': tiled_conf,
                'difference': tiled_conf - regular_conf
            },
            'tiny_objects': {
                'regular': regular_tiny_count,
                'tiled': tiled_tiny_count,
                'difference': tiled_tiny_count - regular_tiny_count
            },
            'average_size': {
                'regular': regular_avg_size,
                'tiled': tiled_avg_size,
                'difference': tiled_avg_size - regular_avg_size
            },
            'timing': {
                'regular': regular_time,
                'tiled': tiled_time,
                'difference': tiled_time - regular_time
            }
        }
        
        return analysis
    
    def print_analysis(self, analysis):
        """Print analysis results in a readable format."""
        print("\n=== Detection Results Analysis ===")
        print(f"Detection counts: Regular={analysis['detection_counts']['regular']}, " 
              f"Tiled={analysis['detection_counts']['tiled']} " 
              f"(Difference: {analysis['detection_counts']['difference']})")
        
        print(f"Average confidence: Regular={analysis['confidence']['regular']:.4f}, " 
              f"Tiled={analysis['confidence']['tiled']:.4f} " 
              f"(Difference: {analysis['confidence']['difference']:.4f})")
        
        print(f"Tiny objects detected: Regular={analysis['tiny_objects']['regular']}, " 
              f"Tiled={analysis['tiny_objects']['tiled']} " 
              f"(Difference: {analysis['tiny_objects']['difference']})")
        
        print(f"Average object size: Regular={analysis['average_size']['regular']:.2f}, " 
              f"Tiled={analysis['average_size']['tiled']:.2f} " 
              f"(Difference: {analysis['average_size']['difference']:.2f})")
        
        print(f"Detection time: Regular={analysis['timing']['regular']:.4f}s, " 
              f"Tiled={analysis['timing']['tiled']:.4f}s " 
              f"(Difference: {analysis['timing']['difference']:.4f}s)")
        
    def process_image(self, image_path, output_dir=None):
        """
        Process a single image with both regular and tiled detection.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save output images (optional)
            
        Returns:
            Dictionary with detection results and analysis
        """
        # Read the image
        print(f"Loading image: {image_path}")
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Error: Could not read image {image_path}")
            # Try reading with PIL as a fallback
            try:
                from PIL import Image
                import numpy as np
                pil_image = Image.open(str(image_path))
                image = np.array(pil_image.convert('RGB'))
                # Convert from RGB to BGR (OpenCV format)
                image = image[:, :, ::-1].copy()
                print(f"Successfully loaded image using PIL: {image.shape}")
            except Exception as e:
                print(f"Also failed with PIL: {e}")
                return None
        else:
            print(f"Successfully loaded image: {image.shape}")
        
        # Perform regular detection
        regular_detections, regular_time = self.detect_regular(image)
        
        # Perform tiled detection
        tiled_detections, tiled_time, coords = self.detect_tiled(image)
        
        # Analyze results
        analysis = self.analyze_results(regular_detections, tiled_detections, regular_time, tiled_time)
        
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{Path(image_path).stem}_comparison.jpg")
            self.visualize_comparison(image, regular_detections, tiled_detections, coords, output_path)
        
        # Return results
        return {
            'image_path': image_path,
            'regular_detections': regular_detections,
            'tiled_detections': tiled_detections,
            'regular_time': regular_time,
            'tiled_time': tiled_time,
            'analysis': analysis
        }
    
    def batch_process(self, image_paths, output_dir=None):
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save output images (optional)
            
        Returns:
            List of results for each image
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for image_path in tqdm(image_paths, desc="Processing images"):
            result = self.process_image(image_path, output_dir)
            if result:
                self.print_analysis(result['analysis'])
                results.append(result)
        
        # Generate summary statistics
        self.generate_summary(results, output_dir)
        
        return results
    
    def generate_summary(self, results, output_dir):
        """
        Generate summary statistics and visualizations for a batch of images.
        
        Args:
            results: List of results from batch_process
            output_dir: Directory to save summary files
        """
        if not results:
            print("No results to summarize")
            return
        
        # Collect data for summary
        detection_diffs = [r['analysis']['detection_counts']['difference'] for r in results]
        conf_diffs = [r['analysis']['confidence']['difference'] for r in results]
        tiny_diffs = [r['analysis']['tiny_objects']['difference'] for r in results]
        size_diffs = [r['analysis']['average_size']['difference'] for r in results]
        time_diffs = [r['analysis']['timing']['difference'] for r in results]
        
        # Calculate summary statistics
        summary = {
            'detection_diff': {
                'mean': np.mean(detection_diffs),
                'median': np.median(detection_diffs),
                'min': np.min(detection_diffs),
                'max': np.max(detection_diffs)
            },
            'conf_diff': {
                'mean': np.mean(conf_diffs),
                'median': np.median(conf_diffs),
                'min': np.min(conf_diffs),
                'max': np.max(conf_diffs)
            },
            'tiny_diff': {
                'mean': np.mean(tiny_diffs),
                'median': np.median(tiny_diffs),
                'min': np.min(tiny_diffs),
                'max': np.max(tiny_diffs)
            },
            'size_diff': {
                'mean': np.mean(size_diffs),
                'median': np.median(size_diffs),
                'min': np.min(size_diffs),
                'max': np.max(size_diffs)
            },
            'time_diff': {
                'mean': np.mean(time_diffs),
                'median': np.median(time_diffs),
                'min': np.min(time_diffs),
                'max': np.max(time_diffs)
            }
        }
        
        # Create summary plot
        plt.figure(figsize=(12, 10))
        
        # Plot detection count differences
        plt.subplot(3, 2, 1)
        plt.hist(detection_diffs, bins=10, color='skyblue', edgecolor='black')
        plt.axvline(summary['detection_diff']['mean'], color='red', linestyle='dashed', linewidth=2)
        plt.title('Detection Count Difference (Tiled - Regular)')
        plt.xlabel('Difference')
        plt.ylabel('Frequency')
        
        # Plot confidence differences
        plt.subplot(3, 2, 2)
        plt.hist(conf_diffs, bins=10, color='skyblue', edgecolor='black')
        plt.axvline(summary['conf_diff']['mean'], color='red', linestyle='dashed', linewidth=2)
        plt.title('Confidence Difference (Tiled - Regular)')
        plt.xlabel('Difference')
        plt.ylabel('Frequency')
        
        # Plot tiny object differences
        plt.subplot(3, 2, 3)
        plt.hist(tiny_diffs, bins=10, color='skyblue', edgecolor='black')
        plt.axvline(summary['tiny_diff']['mean'], color='red', linestyle='dashed', linewidth=2)
        plt.title('Tiny Object Count Difference (Tiled - Regular)')
        plt.xlabel('Difference')
        plt.ylabel('Frequency')
        
        # Plot size differences
        plt.subplot(3, 2, 4)
        plt.hist(size_diffs, bins=10, color='skyblue', edgecolor='black')
        plt.axvline(summary['size_diff']['mean'], color='red', linestyle='dashed', linewidth=2)
        plt.title('Average Size Difference (Tiled - Regular)')
        plt.xlabel('Difference')
        plt.ylabel('Frequency')
        
        # Plot time differences
        plt.subplot(3, 2, 5)
        plt.hist(time_diffs, bins=10, color='skyblue', edgecolor='black')
        plt.axvline(summary['time_diff']['mean'], color='red', linestyle='dashed', linewidth=2)
        plt.title('Time Difference (Tiled - Regular)')
        plt.xlabel('Difference (seconds)')
        plt.ylabel('Frequency')
        
        # Plot comparison summary
        plt.subplot(3, 2, 6)
        categories = ['Detection\nCount', 'Confidence\n(x100)', 'Tiny\nObjects', 'Process\nTime (s)']
        regular_means = [
            np.mean([r['analysis']['detection_counts']['regular'] for r in results]),
            np.mean([r['analysis']['confidence']['regular'] for r in results]) * 100,
            np.mean([r['analysis']['tiny_objects']['regular'] for r in results]),
            np.mean([r['analysis']['timing']['regular'] for r in results])
        ]
        tiled_means = [
            np.mean([r['analysis']['detection_counts']['tiled'] for r in results]),
            np.mean([r['analysis']['confidence']['tiled'] for r in results]) * 100,
            np.mean([r['analysis']['tiny_objects']['tiled'] for r in results]),
            np.mean([r['analysis']['timing']['tiled'] for r in results])
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, regular_means, width, label='Regular', color='red', alpha=0.7)
        plt.bar(x + width/2, tiled_means, width, label='Tiled', color='green', alpha=0.7)
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xticks(x, categories)
        plt.title('Comparison Summary')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'summary_plots.png'))
        
        # Save summary data as CSV
        with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w') as f:
            f.write("=== Tiled Detection vs Regular Detection Summary ===\n\n")
            
            f.write("== Detection Count Difference ==\n")
            f.write(f"Mean: {summary['detection_diff']['mean']:.2f}\n")
            f.write(f"Median: {summary['detection_diff']['median']:.2f}\n")
            f.write(f"Min: {summary['detection_diff']['min']:.2f}\n")
            f.write(f"Max: {summary['detection_diff']['max']:.2f}\n\n")
            
            f.write("== Confidence Difference ==\n")
            f.write(f"Mean: {summary['conf_diff']['mean']:.4f}\n")
            f.write(f"Median: {summary['conf_diff']['median']:.4f}\n")
            f.write(f"Min: {summary['conf_diff']['min']:.4f}\n")
            f.write(f"Max: {summary['conf_diff']['max']:.4f}\n\n")
            
            f.write("== Tiny Object Count Difference ==\n")
            f.write(f"Mean: {summary['tiny_diff']['mean']:.2f}\n")
            f.write(f"Median: {summary['tiny_diff']['median']:.2f}\n")
            f.write(f"Min: {summary['tiny_diff']['min']:.2f}\n")
            f.write(f"Max: {summary['tiny_diff']['max']:.2f}\n\n")
            
            f.write("== Average Size Difference ==\n")
            f.write(f"Mean: {summary['size_diff']['mean']:.2f}\n")
            f.write(f"Median: {summary['size_diff']['median']:.2f}\n")
            f.write(f"Min: {summary['size_diff']['min']:.2f}\n")
            f.write(f"Max: {summary['size_diff']['max']:.2f}\n\n")
            
            f.write("== Time Difference ==\n")
            f.write(f"Mean: {summary['time_diff']['mean']:.4f}\n")
            f.write(f"Median: {summary['time_diff']['median']:.4f}\n")
            f.write(f"Min: {summary['time_diff']['min']:.4f}\n")
            f.write(f"Max: {summary['time_diff']['max']:.4f}\n")
        
        print(f"Summary saved to {output_dir}")

# Utility function for NMS
def torchvision_nms(boxes, scores, iou_threshold):
    """
    Non-maximum suppression using PyTorch's implementation.
    """
    try:
        import torchvision
        return torchvision.ops.nms(boxes, scores, iou_threshold)
    except ImportError:
        # Fall back to manual NMS if torchvision is not available
        keep = []
        idxs = scores.argsort(descending=True)
        while len(idxs) > 0:
            i = idxs[0].item()
            keep.append(torch.tensor(i))
            
            if len(idxs) == 1:
                break
                
            # Compute IoU of the picked box with the rest
            xx1 = torch.max(boxes[i, 0], boxes[idxs[1:], 0])
            yy1 = torch.max(boxes[i, 1], boxes[idxs[1:], 1])
            xx2 = torch.min(boxes[i, 2], boxes[idxs[1:], 2])
            yy2 = torch.min(boxes[i, 3], boxes[idxs[1:], 3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
            
            union = area1 + area2 - inter
            iou = inter / union
            
            # Keep boxes with IoU <= threshold
            mask = (iou <= iou_threshold)
            idxs = idxs[1:][mask]
            
        return torch.stack(keep) if keep else torch.tensor([])

def download_sample_dataset():
    """
    Downloads a sample dataset for tiny object detection if none exists.
    """
    # Create test_images directory if it doesn't exist
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    
    # Check if we already have test images
    existing_images = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
    if len(existing_images) > 0:
        print(f"Using {len(existing_images)} existing test images in {TEST_IMAGES_DIR}")
        return [str(img) for img in existing_images]
    
    # Download sample images from VisDrone
    print("Downloading sample images from VisDrone dataset...")
    # This would normally use urllib or requests to download dataset files
    # For demo purposes, we'll just use some placeholder images
    
    # For demonstration, create a text file with instructions
    with open(TEST_IMAGES_DIR / "DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write("To use this script with real tiny object detection datasets, please download one of the following:\n\n")
        f.write("1. VisDrone Dataset: https://github.com/VisDrone/VisDrone-Dataset\n")
        f.write("2. TinyPerson Dataset: https://github.com/ucas-vg/TinyBenchmark\n\n")
        f.write("Extract the dataset into the 'data/test_images' directory or update the TEST_IMAGES_DIR path in the script.\n")
    
    print("Please download a tiny object detection dataset. See DOWNLOAD_INSTRUCTIONS.txt for details.")
    return []

def main():
    """Main function to demonstrate the tiled detector."""
    # Check for command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Tiled Detector for Tiny Object Detection')
    parser.add_argument('--model', type=str, default='dynamiccompactdetect_finetuned.pt',
                        help='Model name (must be in models directory or provide full path)')
    parser.add_argument('--tile-size', type=int, default=640,
                        help='Size of each square tile (default: 640)')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Overlap between adjacent tiles as a fraction (default: 0.2)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--image-dir', type=str, default='',
                        help='Directory containing test images (default: data/test_images)')
    parser.add_argument('--output-dir', type=str, default='results/tiled_detection',
                        help='Directory to save output images and results (default: results/tiled_detection)')
    
    args = parser.parse_args()
    
    # Set up model path
    if os.path.isfile(args.model):
        model_path = args.model
    else:
        model_path = MODELS_DIR / args.model
        if not os.path.isfile(model_path):
            print(f"Error: Model file not found: {model_path}")
            print(f"Available models in {MODELS_DIR}:")
            for model_file in os.listdir(MODELS_DIR):
                print(f"  - {model_file}")
            return
    
    # Set up image directory
    if args.image_dir:
        image_dir = Path(args.image_dir)
    else:
        image_dir = TEST_IMAGES_DIR
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    
    # Initialize the tiled detector
    detector = TiledDetector(
        model_path=model_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Get list of test images
    if os.path.isdir(image_dir):
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        image_paths = [str(p) for p in image_paths]
    else:
        # Try to download sample dataset
        image_paths = download_sample_dataset()
        
    if not image_paths:
        print("No test images found. Please provide images in the test_images directory.")
        return
    
    print(f"Found {len(image_paths)} test images")
    
    # Process images
    results = detector.batch_process(image_paths, str(output_dir))
    
    print(f"Processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    try:
        import torchvision
    except ImportError:
        print("Warning: torchvision not found. Using manual NMS implementation.")
    
    mp.set_start_method('spawn', force=True)
    main() 