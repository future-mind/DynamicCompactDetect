#!/bin/bash
# Script to clean up old repository directories and consolidate into DynamicCompactDetect

BASE_DIR=$(pwd)
PARENT_DIR=$(dirname "$BASE_DIR")

echo "=== DynamicCompactDetect Repository Cleanup ==="
echo "This script will remove old repository directories to clean up your workspace."
echo "Make sure you've already copied all necessary files to the DynamicCompactDetect directory."
echo ""

# Ask for confirmation
read -p "Are you sure you want to remove old repository directories? This cannot be undone. (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Operation cancelled."
    exit 0
fi

# Check if we're inside the DynamicCompactDetect directory
if [[ "$(basename "$BASE_DIR")" != "DynamicCompactDetect" ]]; then
    echo "Error: This script should be run from within the DynamicCompactDetect directory."
    exit 1
fi

# Move to parent directory to find old directories
cd "$PARENT_DIR"

# Look for old directories
OLD_DIRS=()
if [ -d "DynamicCompact-Detect" ]; then
    OLD_DIRS+=("DynamicCompact-Detect")
fi
if [ -d "DynamicCompact-Detect-Clean" ]; then
    OLD_DIRS+=("DynamicCompact-Detect-Clean")
fi

if [ ${#OLD_DIRS[@]} -eq 0 ]; then
    echo "No old repository directories found to clean up."
    echo "Your workspace is already clean."
    exit 0
fi

echo "Found the following directories to clean up:"
for dir in "${OLD_DIRS[@]}"; do
    echo "- $dir"
done
echo ""

# Optional backup
read -p "Would you like to create a backup archive before deletion? (y/n): " backup
if [ "$backup" = "y" ]; then
    echo "Creating backup archive..."
    BACKUP_FILE="DynamicCompact_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    tar -czf "$BACKUP_FILE" "${OLD_DIRS[@]}"
    echo "Backup created: $BACKUP_FILE"
    echo ""
fi

# Remove old directories
echo "Removing old directories..."
for dir in "${OLD_DIRS[@]}"; do
    echo "Removing $dir..."
    rm -rf "$dir"
done

echo ""
echo "Cleanup complete!"
echo "Your project is now consolidated in the DynamicCompactDetect directory."
echo "You can now proceed with pushing to GitHub using the scripts in the DynamicCompactDetect directory." 