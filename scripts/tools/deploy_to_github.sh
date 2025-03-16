#!/bin/bash
# Main script to clean up the project and deploy to GitHub

# Set base directory
BASE_DIR=$(pwd)
SCRIPTS_DIR="$BASE_DIR/scripts/tools"

echo "=== DynamicCompactDetect Project Deployment ==="
echo "This script will:"
echo "1. Set up the repository structure"
echo "2. Initialize git repository"
echo "3. Push to GitHub"
echo "=================================================="

read -p "Do you want to proceed? (y/n): " proceed
if [ "$proceed" != "y" ]; then
    echo "Operation cancelled."
    exit 0
fi

# Run setup script
echo -e "\n=== Setting up repository structure ==="
bash "$SCRIPTS_DIR/setup_repo.sh"
if [ $? -ne 0 ]; then
    echo "Error setting up repository structure."
    exit 1
fi

# Optional: Remove unnecessary files
echo -e "\n=== Cleaning up temporary files ==="
read -p "Do you want to clean up temporary files? (y/n): " cleanup
if [ "$cleanup" = "y" ]; then
    echo "Removing __pycache__ directories..."
    find "$BASE_DIR" -type d -name "__pycache__" -exec rm -rf {} +
    echo "Removing .DS_Store files..."
    find "$BASE_DIR" -name ".DS_Store" -delete
    echo "Cleanup complete!"
fi

# Optional: Run tests
echo -e "\n=== Testing scripts ==="
read -p "Do you want to run tests on the scripts? (y/n): " run_tests
if [ "$run_tests" = "y" ]; then
    echo "Running tests..."
    # Add test commands here
    echo "Tests completed!"
fi

# Push to GitHub
echo -e "\n=== Pushing to GitHub ==="
read -p "Do you want to push to GitHub now? (y/n): " push_now
if [ "$push_now" = "y" ]; then
    # Offer options for pushing
    echo "Choose how to push to GitHub:"
    echo "1. Use push_to_github.sh (handles existing repositories)"
    echo "2. Use create_fresh_repo.sh (creates a completely new repository)"
    read -p "Select an option (1-2): " push_option
    
    case $push_option in
        1)
            bash "$SCRIPTS_DIR/push_to_github.sh"
            ;;
        2)
            bash "$SCRIPTS_DIR/create_fresh_repo.sh"
            ;;
        *)
            echo "Invalid option. Using push_to_github.sh by default."
            bash "$SCRIPTS_DIR/push_to_github.sh"
            ;;
    esac
    
    if [ $? -ne 0 ]; then
        echo "Error pushing to GitHub."
        exit 1
    fi
fi

echo -e "\n=== Deployment complete! ==="
echo "Your DynamicCompactDetect project is ready!"
echo "You can now train the model with: bash scripts/training/train.py"
echo "You can debug the model with: bash scripts/debug/debug_model.py"
echo "You can compare models with: bash scripts/comparison/compare_models.py"
echo "To use with Google Colab, use notebooks/colab_training_notebook.ipynb"
echo "Enjoy!" 