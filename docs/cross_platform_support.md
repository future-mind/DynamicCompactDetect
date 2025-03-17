# Cross-Platform Support Guide

DynamicCompactDetect (DCD) is designed to work across multiple operating systems and hardware platforms. This guide provides detailed instructions for installing and using DCD on different platforms.

## Supported Platforms

DynamicCompactDetect has been tested and verified on the following platforms:

- **macOS**: 11.0 (Big Sur) and later
- **Linux**: Ubuntu 18.04+, Debian 10+, CentOS 7+
- **Windows**: Windows 10/11 with WSL2 or native Python

## Prerequisites

Before installing DynamicCompactDetect, ensure you have the following prerequisites:

### Common Requirements

- Python 3.8 or later
- pip (Python package installer)
- Git

### Platform-Specific Requirements

#### macOS
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew (recommended): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

#### Linux
- Build essentials: `sudo apt-get install build-essential` (Ubuntu/Debian) or `sudo yum groupinstall "Development Tools"` (CentOS/RHEL)
- Python development headers: `sudo apt-get install python3-dev` (Ubuntu/Debian) or `sudo yum install python3-devel` (CentOS/RHEL)

#### Windows
- Visual C++ Build Tools
- Windows Subsystem for Linux (WSL2) for optimal performance (optional but recommended)

## Installation

### macOS

1. Open Terminal and clone the repository:
   ```bash
   git clone https://github.com/future-mind/dynamiccompactdetect.git
   cd dynamiccompactdetect
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation:
   ```bash
   python tests/test_inference.py
   ```

### Linux

1. Open a terminal and clone the repository:
   ```bash
   git clone https://github.com/future-mind/dynamiccompactdetect.git
   cd dynamiccompactdetect
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make scripts executable:
   ```bash
   chmod +x run_dcd_pipeline.sh
   chmod +x tests/test_inference.py
   ```

5. Verify installation:
   ```bash
   ./tests/test_inference.py
   ```

### Windows (Native Python)

1. Open Command Prompt or PowerShell and clone the repository:
   ```
   git clone https://github.com/future-mind/dynamiccompactdetect.git
   cd dynamiccompactdetect
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Verify installation:
   ```
   python tests\test_inference.py
   ```

### Windows (WSL2)

1. Follow the Linux installation instructions within your WSL2 environment.

## Usage

### Running the Pipeline

#### macOS/Linux

```bash
./run_dcd_pipeline.sh [options]
```

#### Windows (PowerShell)

```powershell
.\run_dcd_pipeline.ps1 [options]
```

### Command-Line Options

The pipeline script accepts the following options:

```
Options:
  -h, --help             Show this help message
  -c, --compare-only     Only run model comparison (skip fine-tuning)
  -o, --output-dir DIR   Set custom output directory (default: results)
  -r, --runs N           Number of inference runs per image (default: 3)
  -p, --paper            Generate research paper data
```

### Python API Usage

You can also use DynamicCompactDetect directly in your Python code:

```python
from ultralytics import YOLO

# Load the model
model = YOLO('models/dynamiccompactdetect_finetuned.pt')

# Run inference
results = model('path/to/image.jpg')

# Process results
for r in results:
    print(f"Detected {len(r.boxes)} objects")
    
    # Access individual detections
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates
        conf = box.conf[0]            # Get confidence score
        cls = int(box.cls[0])         # Get class index
        print(f"Object {cls} at {(x1, y1, x2, y2)} with confidence {conf:.2f}")
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'ultralytics'

This error occurs when the required dependencies are not installed. Make sure you've activated your virtual environment and installed all dependencies:

```bash
pip install -r requirements.txt
```

#### Permission Denied for Shell Scripts

On Unix-like systems (macOS/Linux), you may need to make scripts executable:

```bash
chmod +x run_dcd_pipeline.sh
chmod +x tests/test_inference.py
```

#### CUDA Not Available

If you have an NVIDIA GPU but CUDA is not being detected:

1. Ensure you have the appropriate NVIDIA drivers installed
2. Install the CUDA-enabled version of PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

#### Windows Path Issues

On Windows, you may encounter path-related issues. Try using forward slashes (/) instead of backslashes (\\) in file paths, or use raw strings:

```python
model('C:/path/to/image.jpg')  # or model(r'C:\path\to\image.jpg')
```

## Platform-Specific Optimizations

### macOS

- For Apple Silicon (M1/M2) Macs, the model will automatically use the Metal Performance Shaders (MPS) backend for accelerated inference.
- To explicitly enable MPS acceleration:
  ```python
  import torch
  torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
  ```

### Linux with NVIDIA GPUs

- The model will automatically use CUDA if available.
- To specify a particular GPU:
  ```python
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
  ```

### Windows

- For optimal performance on Windows, consider using WSL2 with the Linux instructions.
- If using native Windows, ensure you have the latest Visual C++ Redistributable installed.

## Getting Help

If you encounter issues not covered in this guide:

1. Check the existing issues on GitHub: https://github.com/future-mind/dynamiccompactdetect/issues
2. Create a new issue with detailed information about your problem
3. Include your operating system, Python version, and any error messages 