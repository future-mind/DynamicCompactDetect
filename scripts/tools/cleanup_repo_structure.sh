#!/bin/bash
# Script to analyze and clean up the repository structure

BASE_DIR=$(pwd)

echo "=== DynamicCompactDetect Repository Structure Cleanup ==="
echo "This script will analyze the repository structure and help you"
echo "remove unnecessary directories and files."
echo ""

# Check if we're inside the DynamicCompactDetect directory
if [[ "$(basename "$BASE_DIR")" != "DynamicCompactDetect" ]]; then
    echo "Error: This script should be run from within the DynamicCompactDetect directory."
    exit 1
fi

# Define the essential directories
ESSENTIAL_DIRS=(
    "src"
    "scripts"
    "configs"
    "data"
    "outputs"
)

# Define useful but non-essential directories
USEFUL_DIRS=(
    "notebooks"
    "tests"
    "docs"
    "datasets"
    "assets"
)

# Create a backup before making changes
echo "Creating backup of current repository state..."
cd ..
BACKUP_FILE="DynamicCompact_repo_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$BACKUP_FILE" DynamicCompactDetect
echo "Backup created: $BACKUP_FILE"
echo ""

# Return to repository directory
cd "$BASE_DIR"

# Analyze repository structure
echo "Analyzing repository structure..."

# Check essential directories
echo "Essential directories:"
for dir in "${ESSENTIAL_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        items=$(find "$dir" -type f | wc -l)
        echo "✓ $dir/ ($items files)"
    else
        echo "✗ $dir/ (MISSING)"
        echo "  Creating $dir directory..."
        mkdir -p "$dir"
    fi
done
echo ""

# Check useful but non-essential directories
echo "Useful but non-essential directories:"
for dir in "${USEFUL_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        items=$(find "$dir" -type f | wc -l)
        if [ "$items" -eq 0 ]; then
            echo "⚠ $dir/ (empty)"
            read -p "  Do you want to keep this empty directory? (y/n): " keep_dir
            if [ "$keep_dir" = "n" ]; then
                rmdir "$dir"
                echo "  Deleted $dir/"
            else
                echo "  Keeping empty $dir/ directory"
                # Create a placeholder file
                touch "$dir/.gitkeep"
            fi
        else
            echo "✓ $dir/ ($items files)"
        fi
    else
        echo "○ $dir/ (not present)"
    fi
done
echo ""

# Check for unnecessary directories
echo "Checking for unnecessary directories..."
all_dirs=$(find . -maxdepth 1 -type d | grep -v "^\.$" | sed 's|^\./||')
for dir in $all_dirs; do
    # Skip dot directories and essential/useful directories
    if [[ "$dir" == .* ]]; then
        continue
    fi
    
    if [[ ! " ${ESSENTIAL_DIRS[@]} " =~ " ${dir} " ]] && [[ ! " ${USEFUL_DIRS[@]} " =~ " ${dir} " ]]; then
        echo "? $dir/ (potentially unnecessary)"
        read -p "  Do you want to keep this directory? (y/n): " keep_dir
        if [ "$keep_dir" = "n" ]; then
            echo "  Removing $dir/..."
            rm -rf "$dir"
        else
            echo "  Keeping $dir/"
        fi
    fi
done
echo ""

# Check scripts directory for organization
echo "Analyzing scripts directory structure..."
if [ -d "scripts" ]; then
    # Expected script subdirectories
    SCRIPT_SUBDIRS=("debug" "training" "evaluation" "comparison")
    
    # Check each expected subdirectory
    for subdir in "${SCRIPT_SUBDIRS[@]}"; do
        if [ -d "scripts/$subdir" ]; then
            items=$(find "scripts/$subdir" -type f | wc -l)
            echo "✓ scripts/$subdir/ ($items files)"
        else
            echo "○ scripts/$subdir/ (not present)"
            read -p "  Create scripts/$subdir/ directory? (y/n): " create_dir
            if [ "$create_dir" = "y" ]; then
                mkdir -p "scripts/$subdir"
                echo "  Created scripts/$subdir/"
            fi
        fi
    done
    
    # Check for loose scripts in scripts directory
    loose_scripts=$(find "scripts" -maxdepth 1 -type f | wc -l)
    echo "ℹ scripts/ has $loose_scripts files directly in the directory"
    
    # Check for other unexpected script subdirectories
    all_script_subdirs=$(find "scripts" -maxdepth 1 -type d | grep -v "^scripts$" | sed 's|^scripts/||')
    for subdir in $all_script_subdirs; do
        if [[ ! " ${SCRIPT_SUBDIRS[@]} " =~ " ${subdir} " ]]; then
            echo "? scripts/$subdir/ (unexpected script subdirectory)"
            read -p "  Do you want to keep this directory? (y/n): " keep_dir
            if [ "$keep_dir" = "n" ]; then
                echo "  Moving files to appropriate directories..."
                # Try to categorize and move files
                for file in $(find "scripts/$subdir" -type f -name "*.py"); do
                    filename=$(basename "$file")
                    if [[ "$filename" == *"debug"* ]]; then
                        mkdir -p "scripts/debug"
                        mv "$file" "scripts/debug/"
                        echo "  Moved $filename to scripts/debug/"
                    elif [[ "$filename" == *"train"* ]]; then
                        mkdir -p "scripts/training"
                        mv "$file" "scripts/training/"
                        echo "  Moved $filename to scripts/training/"
                    elif [[ "$filename" == *"eval"* ]]; then
                        mkdir -p "scripts/evaluation"
                        mv "$file" "scripts/evaluation/"
                        echo "  Moved $filename to scripts/evaluation/"
                    elif [[ "$filename" == *"compare"* ]] || [[ "$filename" == *"comparison"* ]]; then
                        mkdir -p "scripts/comparison"
                        mv "$file" "scripts/comparison/"
                        echo "  Moved $filename to scripts/comparison/"
                    else
                        mv "$file" "scripts/"
                        echo "  Moved $filename to scripts/"
                    fi
                done
                
                # Remove the now-empty directory
                rmdir "scripts/$subdir" 2>/dev/null
                if [ $? -eq 0 ]; then
                    echo "  Removed scripts/$subdir/"
                else
                    echo "  Warning: scripts/$subdir/ may still contain files"
                    read -p "  Force remove directory? (y/n): " force_remove
                    if [ "$force_remove" = "y" ]; then
                        rm -rf "scripts/$subdir"
                        echo "  Removed scripts/$subdir/"
                    fi
                fi
            fi
        fi
    done
fi
echo ""

# Check src directory for organization
echo "Analyzing src directory structure..."
if [ -d "src" ]; then
    # Expected src subdirectories
    SRC_SUBDIRS=("models" "utils" "data")
    
    # Check each expected subdirectory
    for subdir in "${SRC_SUBDIRS[@]}"; do
        if [ -d "src/$subdir" ]; then
            items=$(find "src/$subdir" -type f | wc -l)
            echo "✓ src/$subdir/ ($items files)"
        else
            echo "○ src/$subdir/ (not present)"
            read -p "  Create src/$subdir/ directory? (y/n): " create_dir
            if [ "$create_dir" = "y" ]; then
                mkdir -p "src/$subdir"
                echo "  Created src/$subdir/"
            fi
        fi
    done
    
    # Check for unexpected src subdirectories
    all_src_subdirs=$(find "src" -maxdepth 1 -type d | grep -v "^src$" | sed 's|^src/||')
    for subdir in $all_src_subdirs; do
        if [[ ! " ${SRC_SUBDIRS[@]} " =~ " ${subdir} " ]]; then
            echo "? src/$subdir/ (unexpected source subdirectory)"
            read -p "  Do you want to keep this directory? (y/n): " keep_dir
            if [ "$keep_dir" = "n" ]; then
                echo "  Removing src/$subdir/..."
                rm -rf "src/$subdir"
            fi
        fi
    done
fi
echo ""

# Check for duplicate files
echo "Checking for duplicate requirement files..."
if [ -f "requirements.original.txt" ] && [ -f "requirements.txt" ]; then
    echo "Found both requirements.txt and requirements.original.txt"
    read -p "Do you want to keep the original requirements file? (y/n): " keep_original
    if [ "$keep_original" = "n" ]; then
        rm requirements.original.txt
        echo "Deleted requirements.original.txt"
    fi
fi
echo ""

# Check for colab_training.ipynb file
if [ -f "colab_training.ipynb" ]; then
    file_size=$(stat -f%z "colab_training.ipynb")
    if [ "$file_size" -lt 10 ]; then
        echo "colab_training.ipynb is empty or very small ($file_size bytes)"
        read -p "Do you want to delete this file? (y/n): " delete_file
        if [ "$delete_file" = "y" ]; then
            rm colab_training.ipynb
            echo "Deleted colab_training.ipynb"
        fi
    else
        echo "colab_training.ipynb is properly populated ($file_size bytes)"
    fi
fi
echo ""

echo "Repository structure cleanup complete!"
echo ""
echo "Next steps:"
echo "1. Review the final directory structure"
echo "2. Make sure all files are in their correct places"
echo "3. Run ./deploy_to_github.sh to deploy to GitHub" 