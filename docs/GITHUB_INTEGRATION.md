# GitHub Integration Guide

This document provides instructions for pushing the DynamicCompactDetect project to GitHub and verifying it works correctly when cloned fresh.

## Pushing to GitHub

### 1. Create a new GitHub repository

1. Go to [GitHub](https://github.com) and log in to your account
2. Click the "+" button in the top right and select "New repository"
3. Name your repository `dynamiccompactdetect`
4. Add a description: "A lightweight yet powerful object detection model optimized for edge devices"
5. Choose "Public" visibility
6. Do not initialize with a README, .gitignore, or license (we'll push our existing ones)
7. Click "Create repository"

### 2. Initialize Git and push the project

From your local DynamicCompactDetect directory, run:

```bash
# Initialize Git repository (if not already done)
git init

# Add all files to staging
git add .

# Commit the changes
git commit -m "Initial commit of DynamicCompactDetect project"

# Add the remote repository
git remote add origin https://github.com/future-mind/dynamiccompactdetect.git

# Push to GitHub
git push -u origin main
```

Replace `future-mind` with your actual GitHub username.

## Cloning and Testing the Repository

You can use our end-to-end test script to verify that the repository works correctly when cloned fresh:

```bash
tests/run_end_to_end_test.sh
```

This script will:
1. Clone the repository to a temporary directory
2. Set up a virtual environment
3. Install dependencies
4. Run basic model tests
5. Execute the complete pipeline
6. Generate research paper data
7. Verify that all results are correctly generated

### Manual Verification

If you prefer to test manually:

1. Clone the repository:
   ```bash
   git clone https://github.com/future-mind/dynamiccompactdetect.git test-dcd
   cd test-dcd
   ```

2. Create and activate a virtual environment:
   ```bash
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the pipeline:
   ```bash
   ./run_dcd_pipeline.sh --compare-only
   ```

5. Generate research paper data:
   ```bash
   ./run_dcd_pipeline.sh --paper
   ```

6. Check the results:
   ```bash
   ls -la results/comparisons
   ls -la results/research_paper
   ```

## Troubleshooting

### Common Issues

1. **Missing model files**: Ensure the model files are available in the `models/` directory.
   ```bash
   ls -la models/
   ```

2. **Permission denied for scripts**: Make sure shell scripts are executable.
   ```bash
   chmod +x run_dcd_pipeline.sh
   chmod +x tests/run_end_to_end_test.sh
   ```

3. **Missing test images**: The pipeline script should automatically download test images, but if it fails:
   ```bash
   mkdir -p data/test_images
   curl -L -o data/test_images/zidane.jpg https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg
   curl -L -o data/test_images/bus.jpg https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg
   ```

4. **Virtual environment issues**: If you encounter problems with the virtual environment:
   ```bash
   # Delete and recreate
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

### Reporting Issues

If you encounter persistent issues:

1. Check the existing issues on GitHub: https://github.com/future-mind/dynamiccompactdetect/issues
2. Create a new issue with detailed information about the problem
3. Include your operating system, Python version, and any error messages 