#!/bin/bash
# Run model comparison for DynamicCompact-Detect

# Set the base directory
BASE_DIR=$(dirname "$(dirname "$(dirname "$0")")")
cd $BASE_DIR

# Ensure output directories exist
mkdir -p outputs/results/comparison

# Check if we have a virtual environment and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if required models exist
MODEL_DIR="outputs/checkpoints"
mkdir -p $MODEL_DIR

DYNAMICCOMPACT_MODEL="$MODEL_DIR/dynamiccompact.pt"
YOLOV10_MODEL="$MODEL_DIR/yolov10n.pt"
RTDETR_MODEL="$MODEL_DIR/rtdetr-l.pt"

# Check if DynamicCompact model exists
if [ ! -f "$DYNAMICCOMPACT_MODEL" ]; then
    echo "DynamicCompact model not found at $DYNAMICCOMPACT_MODEL"
    echo "Checking for it in runs/train_minimal/best_model.pt..."
    
    if [ -f "runs/train_minimal/best_model.pt" ]; then
        echo "Found model in runs/train_minimal. Copying to $MODEL_DIR..."
        cp runs/train_minimal/best_model.pt $DYNAMICCOMPACT_MODEL
    else
        echo "Warning: DynamicCompact model not found. Comparison may fail."
    fi
fi

# Check if YOLOv10 model exists
if [ ! -f "$YOLOV10_MODEL" ]; then
    echo "YOLOv10 model not found at $YOLOV10_MODEL"
    echo "Checking for it in the project root..."
    
    if [ -f "yolov8n.pt" ]; then
        echo "Found model in project root. Copying to $MODEL_DIR..."
        cp yolov8n.pt $YOLOV10_MODEL
    else
        echo "Warning: YOLOv10 model not found. Comparison may fail for this model."
    fi
fi

# Check if RT-DETR model exists
if [ ! -f "$RTDETR_MODEL" ]; then
    echo "RT-DETR model not found at $RTDETR_MODEL"
    echo "Checking for it in the project root..."
    
    if [ -f "rtdetr-l.pt" ]; then
        echo "Found model in project root. Copying to $MODEL_DIR..."
        cp rtdetr-l.pt $RTDETR_MODEL
    else
        echo "Warning: RT-DETR model not found. Comparison may fail for this model."
    fi
fi

# Parse command line arguments
DATASET_PATH="datasets/coco/val2017"
NUM_IMAGES=50
CONF_THRES=0.25
DEVICE="cpu"

# Check for CUDA availability
if command -v nvidia-smi &> /dev/null; then
    DEVICE=0
    echo "CUDA detected. Using GPU for comparison."
fi

# Process arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dataset)
        DATASET_PATH="$2"
        shift
        shift
        ;;
        --num-imgs)
        NUM_IMAGES="$2"
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

echo "===================================================="
echo "Running model comparison with the following settings:"
echo "  Dataset path: $DATASET_PATH"
echo "  Number of images: $NUM_IMAGES"
echo "  Confidence threshold: $CONF_THRES"
echo "  Device: $DEVICE"
echo "===================================================="

# Run the comparison script
python3 scripts/comparison/compare_models.py \
    --dataset $DATASET_PATH \
    --num-imgs $NUM_IMAGES \
    --conf-thres $CONF_THRES \
    --device $DEVICE

echo "Comparison complete. Results saved to outputs/results/comparison/" 