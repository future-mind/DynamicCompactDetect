#!/bin/bash
# Full training and evaluation pipeline for DynamicCompactDetect on COCO dataset

# Exit on error
set -e

# Configuration
CONFIG_PATH="config/full_coco_config.yaml"
DATA_DIR="data/coco"
TRAIN_OUTPUT_DIR="results/full_coco"
COMPARISON_OUTPUT_DIR="results/full_dataset_comparison"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda:0"
    echo "CUDA detected, using GPU: $DEVICE"
else
    DEVICE="cpu"
    echo "CUDA not detected, using CPU"
fi

# Create directories
mkdir -p $DATA_DIR
mkdir -p $TRAIN_OUTPUT_DIR
mkdir -p $COMPARISON_OUTPUT_DIR

# Step 1: Check and Download COCO dataset if needed
echo "=== Step 1: Checking COCO dataset ==="

# Function to check if full dataset exists
check_full_dataset() {
    if [ -d "$DATA_DIR/images/train2017" ] && [ -d "$DATA_DIR/images/val2017" ] && [ -d "$DATA_DIR/annotations" ]; then
        echo "Full COCO dataset found at $DATA_DIR"
        return 0
    else
        echo "Full COCO dataset not found"
        return 1
    fi
}

# Function to check if mini dataset exists
check_mini_dataset() {
    if [ -d "$DATA_DIR/images/mini_train2017" ] && [ -d "$DATA_DIR/images/mini_val2017" ] && [ -d "$DATA_DIR/annotations" ]; then
        echo "Mini COCO dataset found at $DATA_DIR"
        return 0
    else
        echo "Mini COCO dataset not found"
        return 1
    fi
}

# Ask user if they want full dataset or mini dataset
read -p "Use full COCO dataset? (y/n, n will use mini COCO for testing): " use_full

if [ "$use_full" = "y" ]; then
    # Check if full dataset exists
    if check_full_dataset; then
        echo "Using existing full COCO dataset"
    else
        echo "Downloading full COCO dataset. This might take a while depending on your internet connection..."
        python data/dataset_utils.py download --dataset coco --data-dir $DATA_DIR --year 2017
    fi
else
    # Check if mini dataset exists
    if check_mini_dataset; then
        echo "Using existing mini COCO dataset"
    else
        echo "Downloading mini COCO dataset..."
        python data/dataset_utils.py download --dataset coco128 --data-dir $DATA_DIR
    fi
    
    # Use mini config
    CONFIG_PATH="config/mini_coco_config.yaml"
    # Copy full config and modify paths
    cp config/full_coco_config.yaml $CONFIG_PATH
    # Update paths to mini dataset
    sed -i 's/train2017/mini_train2017/g' $CONFIG_PATH
    sed -i 's/val2017/mini_val2017/g' $CONFIG_PATH
fi

# Step 2: Train the model
echo ""
echo "=== Step 2: Training DynamicCompactDetect ==="
python train/train_full_coco.py \
    --config $CONFIG_PATH \
    --output-dir $TRAIN_OUTPUT_DIR

# Step 3: Run comparison with YOLO models
echo ""
echo "=== Step 3: Comparing with YOLO models ==="
python eval/compare_with_yolo.py \
    --config $CONFIG_PATH \
    --dcd-weights $TRAIN_OUTPUT_DIR/weights/best_model.pth \
    --output-dir $COMPARISON_OUTPUT_DIR \
    --input-sizes 320x320 640x640 \
    --iterations 50 \
    --num-samples 20 \
    --benchmark-only

# Step 4: Print summary
echo ""
echo "=== Training and Evaluation Completed ==="
echo "Results can be found in:"
echo "- Training results: $TRAIN_OUTPUT_DIR"
echo "- Comparison results: $COMPARISON_OUTPUT_DIR"
echo ""
echo "To view the comparison charts:"
echo "- Benchmark results: $COMPARISON_OUTPUT_DIR/benchmark_results.json"
echo "- FPS comparison: $COMPARISON_OUTPUT_DIR/fps_comparison.png"
echo "- Size comparison: $COMPARISON_OUTPUT_DIR/size_comparison.png"
echo "- Efficiency comparison: $COMPARISON_OUTPUT_DIR/efficiency_comparison.png"
echo "- Comprehensive comparison: $COMPARISON_OUTPUT_DIR/comprehensive_comparison.png"
echo "- Detection comparisons: $COMPARISON_OUTPUT_DIR/detection_comparisons/" 