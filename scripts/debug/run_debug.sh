#!/bin/bash
# Run debug analysis on DynamicCompact-Detect model

# Set the base directory
BASE_DIR=$(dirname "$(dirname "$(dirname "$0")")")
cd $BASE_DIR

# Ensure output directories exist
mkdir -p outputs/results/debug/feature_maps

# Check if we have a virtual environment and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set default parameters
WEIGHTS="outputs/checkpoints/dynamiccompact.pt"
IMG_PATH="datasets/coco/val2017/000000119445.jpg"
CONF_THRES=0.001
DEVICE="cpu"

# Check for CUDA availability
if command -v nvidia-smi &> /dev/null; then
    DEVICE=0
    echo "CUDA detected. Using GPU for debugging."
fi

# Process arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --weights)
        WEIGHTS="$2"
        shift
        shift
        ;;
        --img-path)
        IMG_PATH="$2"
        shift
        shift
        ;;
        --conf-thres)
        CONF_THRES="$2"
        shift
        shift
        ;;
        --device)
        DEVICE="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        shift
        ;;
    esac
done

# Check if weights file exists
if [ ! -f "$WEIGHTS" ]; then
    # Try to find it in runs directory
    if [ -f "runs/train_minimal/best_model.pt" ]; then
        echo "Using weights from runs/train_minimal/best_model.pt"
        WEIGHTS="runs/train_minimal/best_model.pt"
    else
        echo "Error: Weights file not found. Please specify a valid path with --weights"
        exit 1
    fi
fi

# Check if image file exists
if [ ! -f "$IMG_PATH" ]; then
    echo "Error: Image file not found. Please specify a valid path with --img-path"
    exit 1
fi

echo "===================================================="
echo "Running model debug with the following settings:"
echo "  Weights: $WEIGHTS"
echo "  Image: $IMG_PATH"
echo "  Confidence threshold: $CONF_THRES"
echo "  Device: $DEVICE"
echo "===================================================="

# Run the debug script
python3 scripts/debug/debug_model.py \
    --weights $WEIGHTS \
    --img-path $IMG_PATH \
    --conf-thres $CONF_THRES \
    --device $DEVICE

echo "Debug analysis complete. Results saved to outputs/results/debug/" 