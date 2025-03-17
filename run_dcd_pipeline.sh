#!/bin/bash

# DynamicCompactDetect Pipeline Script
# This script runs the complete pipeline for model comparison and evaluation

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="linux"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    OS_TYPE="windows"
else
    OS_TYPE="unknown"
    echo "Warning: Unknown OS type. Some features may not work correctly."
fi

set -e  # Exit on error

# Display help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "DynamicCompactDetect (DCD) Pipeline Script"
    echo ""
    echo "Options:"
    echo "  -h, --help             Show this help message"
    echo "  -c, --compare-only     Only run model comparison (skip fine-tuning)"
    echo "  -o, --output-dir DIR   Set custom output directory (default: results)"
    echo "  -r, --runs N           Number of inference runs per image (default: 3)"
    echo "  -p, --paper            Generate research paper data"
    echo ""
    echo "Examples:"
    echo "  $0                     Run the complete pipeline"
    echo "  $0 --compare-only      Only run model comparison"
    echo "  $0 --output-dir custom_results --runs 5"
    echo "  $0 --paper             Generate research paper data"
    echo ""
}

# Default values
COMPARE_ONLY=false
OUTPUT_DIR="results"
NUM_RUNS=3
GENERATE_PAPER=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--compare-only)
            COMPARE_ONLY=true
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -p|--paper)
            GENERATE_PAPER=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "DynamicCompactDetect (DCD) Pipeline"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Number of inference runs: $NUM_RUNS"
echo "Compare only mode: $COMPARE_ONLY"
echo "Generate paper data: $GENERATE_PAPER"
echo "OS: $OS_TYPE"
echo "=============================================="

# OS-specific setup
if [ "$OS_TYPE" = "windows" ]; then
    echo "Setting up for Windows environment..."
    # Check if we're in a Python virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Checking for Python environment..."
        if [ -d "venv" ]; then
            echo "Found 'venv' directory. Activating..."
            # Windows-style activation
            source venv/Scripts/activate
        else
            echo "Warning: No virtual environment found. Consider creating one with:"
            echo "python -m venv venv"
            echo "venv\\Scripts\\activate"
        fi
    fi
elif [ "$OS_TYPE" = "linux" ]; then
    echo "Setting up for Linux environment..."
    # Check if we're in a Python virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Checking for Python environment..."
        if [ -d "venv" ]; then
            echo "Found 'venv' directory. Activating..."
            source venv/bin/activate
        else
            echo "Warning: No virtual environment found. Consider creating one with:"
            echo "python3 -m venv venv"
            echo "source venv/bin/activate"
        fi
    fi
elif [ "$OS_TYPE" = "macos" ]; then
    echo "Setting up for macOS environment..."
    # Check if we're in a Python virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Checking for Python environment..."
        if [ -d "venv" ]; then
            echo "Found 'venv' directory. Activating..."
            source venv/bin/activate
        else
            echo "Warning: No virtual environment found. Consider creating one with:"
            echo "python3 -m venv venv"
            echo "source venv/bin/activate"
        fi
    fi
fi

# Check for required files and directories
echo "Checking project structure..."
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
fi

if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir -p data/test_images
fi

# Check for model files
echo "Checking for model files..."
MISSING_MODELS=false
MODEL_FILES=("models/yolov8n.pt" "models/dynamiccompactdetect_finetuned.pt")

for MODEL_FILE in "${MODEL_FILES[@]}"; do
    if [ ! -f "$MODEL_FILE" ]; then
        echo "- Missing: $MODEL_FILE"
        MISSING_MODELS=true
    else
        echo "- Found: $MODEL_FILE"
    fi
done

if [ "$MISSING_MODELS" = true ]; then
    echo "Please ensure all model files are available in the models directory."
    echo "You can download them from the project repository or run the setup script."
    exit 1
fi

# Check for test images
echo "Checking for test images..."
if [ ! "$(ls -A data/test_images 2>/dev/null)" ]; then
    echo "- No test images found. Downloading sample images..."
    # Use appropriate method based on OS
    if command -v curl &> /dev/null; then
        echo "  Using curl to download images..."
        curl -L -o data/test_images/zidane.jpg https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg
        curl -L -o data/test_images/bus.jpg https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg
    elif command -v wget &> /dev/null; then
        echo "  Using wget to download images..."
        wget -P data/test_images https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg
        wget -P data/test_images https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg
    else
        echo "  Error: Neither curl nor wget found. Please download test images manually."
        exit 1
    fi
else
    echo "- Test images found in data/test_images"
fi

# Run fine-tuning if not in compare-only mode
if [ "$COMPARE_ONLY" = false ]; then
    echo "=============================================="
    echo "Running DynamicCompactDetect fine-tuning..."
    echo "=============================================="
    echo "This step is skipped for now - please run manually if needed."
    # Uncomment the line below when fine-tuning script is ready
    # python scripts/finetune_dynamiccompactdetect.py --output-dir "$OUTPUT_DIR/finetune"
fi

# Run model comparison
echo "=============================================="
echo "Running model comparison..."
echo "=============================================="
python scripts/compare_models.py --num-runs "$NUM_RUNS" --output-dir "$OUTPUT_DIR/comparisons"

# Generate research paper data if requested
if [ "$GENERATE_PAPER" = true ]; then
    echo "=============================================="
    echo "Generating research paper data..."
    echo "=============================================="
    python scripts/generate_paper_data.py --output-dir "$OUTPUT_DIR"
    
    echo "Research paper materials generated in: $OUTPUT_DIR/research_paper"
fi

echo "=============================================="
echo "Pipeline completed successfully!"
echo "=============================================="
echo "Results are available in: $OUTPUT_DIR"
echo "" 