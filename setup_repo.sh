#!/bin/bash
# Script to set up the DynamicCompactDetect repository structure

# Set the current directory as the base
BASE_DIR=$(pwd)
ORIGINAL_DIR="$BASE_DIR/../DynamicCompact-Detect"
CLEAN_DIR="$BASE_DIR"

# Check if original directory exists
if [ ! -d "$ORIGINAL_DIR" ]; then
    echo "Error: Original directory not found at $ORIGINAL_DIR"
    echo "Looking for alternative directory..."
    
    # Check for alternative directory paths
    ALT_DIR1="$BASE_DIR/../DynamicCompactDetect-Original"
    ALT_DIR2="$BASE_DIR/../../DynamicCompact-Detect"
    
    if [ -d "$ALT_DIR1" ]; then
        ORIGINAL_DIR="$ALT_DIR1"
        echo "Found original directory at $ORIGINAL_DIR"
    elif [ -d "$ALT_DIR2" ]; then
        ORIGINAL_DIR="$ALT_DIR2"
        echo "Found original directory at $ORIGINAL_DIR"
    else
        echo "No source directory found. You may need to specify the source directory manually."
        read -p "Continue with repository setup without source files? (y/n): " continue_setup
        if [ "$continue_setup" != "y" ]; then
            exit 1
        fi
    fi
fi

# Create directories that don't already exist
mkdir -p src/{models,utils,data}
mkdir -p scripts/{debug,training,evaluation,comparison}
mkdir -p configs/{model_configs,train_configs}
mkdir -p notebooks tests docs assets
mkdir -p data outputs/{runs,checkpoints,results}

# Create placeholder files for version control in empty directories
touch outputs/runs/.gitkeep
touch outputs/checkpoints/.gitkeep
touch outputs/results/.gitkeep
touch data/.gitkeep
touch notebooks/.gitkeep
touch tests/.gitkeep
touch docs/.gitkeep
touch assets/.gitkeep

# Copy model files if original directory exists
if [ -d "$ORIGINAL_DIR/models" ]; then
    echo "Copying model files..."
    cp -r "$ORIGINAL_DIR/models/"* src/models/
fi

# Copy utility files
if [ -d "$ORIGINAL_DIR/utils" ]; then
    echo "Copying utility files..."
    cp -r "$ORIGINAL_DIR/utils/"* src/utils/
fi

# Copy configuration files
if [ -d "$ORIGINAL_DIR/configs" ]; then
    echo "Copying configuration files..."
    cp -r "$ORIGINAL_DIR/configs/"* configs/model_configs/
fi

# Move selected debug scripts
if [ -f "$ORIGINAL_DIR/model_diagnosis.py" ]; then
    echo "Moving debug scripts..."
    cp "$ORIGINAL_DIR/model_diagnosis.py" scripts/debug/
fi

# Move any useful debugging scripts
for script in "$ORIGINAL_DIR/debug_"*.py; do
    if [ -f "$script" ]; then
        echo "Moving $script to scripts/debug/"
        cp "$script" scripts/debug/
    fi
done

# Move training scripts
if [ -f "$ORIGINAL_DIR/train_minimal.py" ]; then
    echo "Moving training scripts..."
    cp "$ORIGINAL_DIR/train_minimal.py" scripts/training/train.py
fi

# Move inference scripts
if [ -f "$ORIGINAL_DIR/detect.py" ]; then
    echo "Moving inference scripts..."
    cp "$ORIGINAL_DIR/detect.py" scripts/inference.py
fi
if [ -f "$ORIGINAL_DIR/simple_inference.py" ]; then
    cp "$ORIGINAL_DIR/simple_inference.py" scripts/inference_simple.py
fi

# Move evaluation scripts
if [ -f "$ORIGINAL_DIR/evaluate_model.py" ]; then
    echo "Moving evaluation scripts..."
    cp "$ORIGINAL_DIR/evaluate_model.py" scripts/evaluation/
fi

# Move comparison scripts
if [ -f "$ORIGINAL_DIR/compare_models.py" ]; then
    echo "Moving comparison scripts..."
    cp "$ORIGINAL_DIR/compare_models.py" scripts/comparison/
fi
if [ -f "$ORIGINAL_DIR/real_model_comparison.py" ]; then
    cp "$ORIGINAL_DIR/real_model_comparison.py" scripts/comparison/
fi
if [ -f "$ORIGINAL_DIR/visual_model_comparison.py" ]; then
    cp "$ORIGINAL_DIR/visual_model_comparison.py" scripts/comparison/
fi

# Copy README if it exists
if [ -f "$ORIGINAL_DIR/README.md" ]; then
    echo "Copying original README.md for reference..."
    cp "$ORIGINAL_DIR/README.md" docs/original_README.md
fi

# Copy requirement files
if [ -f "$ORIGINAL_DIR/requirements.txt" ]; then
    echo "Copying requirements.txt..."
    cp "$ORIGINAL_DIR/requirements.txt" requirements.original.txt
fi

# Make scripts executable
chmod +x scripts/debug/*.py
chmod +x scripts/training/*.py
chmod +x scripts/evaluation/*.py
chmod +x scripts/comparison/*.py
chmod +x scripts/*.py
chmod +x scripts/debug/run_debug.sh
chmod +x scripts/comparison/run_comparison.sh

echo "Repository structure setup complete!"
echo "Next steps:"
echo "1. Review and test copied files"
echo "2. Update import paths in copied scripts to match new structure"
echo "3. Initialize git repository and commit changes"
echo ""
echo "To initialize git repository:"
echo "git init"
echo "git add ."
echo "git commit -m \"Initial commit of DynamicCompactDetect\""
echo "git remote add origin https://github.com/future-mind/DynamicCompactDetect.git"
echo "git push -u origin main" 