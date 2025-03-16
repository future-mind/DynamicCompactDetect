#!/bin/bash
# Script to clean up the workspace and consolidate all scripts into the repository

BASE_DIR=$(pwd)
PARENT_DIR=$(dirname "$BASE_DIR")

echo "=== DynamicCompactDetect Workspace Cleanup ==="
echo "This script will:"
echo "1. Move useful loose script files into the repository"
echo "2. Remove duplicated script files"
echo "3. Delete old repository directories"
echo "4. Clean up unnecessary directories inside the repository"
echo ""

# Check if we're inside the DynamicCompactDetect directory
if [[ "$(basename "$BASE_DIR")" != "DynamicCompactDetect" ]]; then
    echo "Error: This script should be run from within the DynamicCompactDetect directory."
    exit 1
fi

# Create backup of workspace before making changes
echo "Creating backup of current workspace state..."
cd "$PARENT_DIR"
BACKUP_FILE="DynamicCompact_workspace_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$BACKUP_FILE" --exclude="*.tar.gz" .
echo "Backup created: $BACKUP_FILE"
echo ""

# Move back to the repository directory
cd "$BASE_DIR"

# Check and organize loose script files
echo "Checking for loose script files to organize..."
cd "$PARENT_DIR"

# Array of script files to be processed
SCRIPT_FILES=(
    "debug_script.py"
    "simple_inference.py"
    "analyze_model_output.py"
    "debug_model_output.py"
    "debug_simple.py"
    "debug_inference.py"
    "real_model_comparison.py"
    "debug_model.py"
    "debug_dynamiccompact.py"
    "detect_minimal.py"
)

# Process each script file
for script in "${SCRIPT_FILES[@]}"; do
    if [ -f "$script" ]; then
        echo "Found $script"
        
        # Determine destination directory based on filename
        if [[ "$script" == *"debug"* ]] || [[ "$script" == *"analyze"* ]]; then
            dest_dir="$BASE_DIR/scripts/debug"
            mkdir -p "$dest_dir"
            cp "$script" "$dest_dir/"
            echo "  → Copied to scripts/debug/"
        elif [[ "$script" == *"inference"* ]] || [[ "$script" == *"detect"* ]]; then
            dest_dir="$BASE_DIR/scripts"
            cp "$script" "$dest_dir/"
            echo "  → Copied to scripts/"
        elif [[ "$script" == *"comparison"* ]]; then
            dest_dir="$BASE_DIR/scripts/comparison"
            mkdir -p "$dest_dir"
            cp "$script" "$dest_dir/"
            echo "  → Copied to scripts/comparison/"
        else
            dest_dir="$BASE_DIR/scripts"
            cp "$script" "$dest_dir/"
            echo "  → Copied to scripts/ (general purpose)"
        fi
    fi
done

# Check if user wants to remove loose scripts after copying
echo ""
read -p "Do you want to delete the loose script files from the workspace? (y/n): " delete_loose
if [ "$delete_loose" = "y" ]; then
    echo "Deleting loose script files..."
    for script in "${SCRIPT_FILES[@]}"; do
        if [ -f "$script" ]; then
            rm "$script"
            echo "  Deleted $script"
        fi
    done
fi

# Clean up old repository directories
echo ""
echo "Checking for old repository directories..."
OLD_DIRS=()
if [ -d "DynamicCompact-Detect" ]; then
    OLD_DIRS+=("DynamicCompact-Detect")
fi
if [ -d "DynamicCompact-Detect-Clean" ]; then
    OLD_DIRS+=("DynamicCompact-Detect-Clean")
fi

if [ ${#OLD_DIRS[@]} -eq 0 ]; then
    echo "No old repository directories found to clean up."
else
    echo "Found the following old directories:"
    for dir in "${OLD_DIRS[@]}"; do
        echo "- $dir"
    done
    
    read -p "Do you want to delete these old repository directories? (y/n): " delete_old
    if [ "$delete_old" = "y" ]; then
        for dir in "${OLD_DIRS[@]}"; do
            echo "Removing $dir..."
            rm -rf "$dir"
        done
        echo "Old directories deleted."
    fi
fi

# Check for other potentially unnecessary directories
echo ""
echo "Checking for other potentially unnecessary directories..."

if [ -d "RT-DETR" ]; then
    echo "Found RT-DETR directory."
    read -p "Do you want to keep the RT-DETR directory for comparison purposes? (y/n): " keep_rtdetr
    if [ "$keep_rtdetr" = "n" ]; then
        echo "Removing RT-DETR directory..."
        rm -rf "RT-DETR"
    fi
fi

if [ -d "yolov10" ]; then
    echo "Found yolov10 directory."
    read -p "Do you want to keep the yolov10 directory for comparison purposes? (y/n): " keep_yolo
    if [ "$keep_yolo" = "n" ]; then
        echo "Removing yolov10 directory..."
        rm -rf "yolov10"
    fi
fi

# Now clean up unnecessary directories inside the repository
echo ""
echo "Cleaning up unnecessary directories inside the repository..."
cd "$BASE_DIR"

# Check for empty directories and offer to delete them
find . -type d -empty -not -path "*/\.*" | while read -r dir; do
    if [[ "$dir" != "./outputs/runs" && "$dir" != "./outputs/checkpoints" && "$dir" != "./outputs/results" ]]; then
        echo "Empty directory found: $dir"
        read -p "Do you want to delete this empty directory? (y/n): " delete_empty
        if [ "$delete_empty" = "y" ]; then
            rmdir "$dir"
            echo "  Deleted $dir"
        fi
    fi
done

# Check for duplicate requirements files
if [ -f "requirements.original.txt" ] && [ -f "requirements.txt" ]; then
    echo ""
    echo "Found both requirements.txt and requirements.original.txt"
    read -p "Do you want to keep the original requirements file? (y/n): " keep_original
    if [ "$keep_original" = "n" ]; then
        rm requirements.original.txt
        echo "  Deleted requirements.original.txt"
    fi
fi

# Check and clean up colab_training files
if [ -f "colab_training.ipynb" ] && [ $(wc -c < "colab_training.ipynb") -lt 10 ]; then
    echo ""
    echo "colab_training.ipynb appears to be empty or very small."
    read -p "Do you want to delete this likely empty notebook file? (y/n): " delete_notebook
    if [ "$delete_notebook" = "y" ]; then
        rm colab_training.ipynb
        echo "  Deleted colab_training.ipynb"
    fi
fi

echo ""
echo "Workspace cleanup complete!"
echo "All important scripts are now organized inside the DynamicCompactDetect repository."
echo ""
echo "Next steps:"
echo "1. Make sure the scripts have the correct imports/paths"
echo "2. Run ./deploy_to_github.sh to deploy to GitHub"
echo "3. Delete this cleanup script if no longer needed" 