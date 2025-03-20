# Script Standardization Documentation

## Overview

This document explains the standardization of dataset utility scripts in the DynamicCompactDetect project. We've unified multiple scripts that were performing similar functions into a single, comprehensive utility module.

## Changes Made

1. **Unified Dataset Utilities:**
   - Created a comprehensive `data/dataset_utils.py` module that combines functionality from:
     - `scripts/download_coco.py`
     - `data/download_dataset.py`
   
2. **Deprecated Old Scripts:**
   - Moved original scripts to deprecated folders
   - Created backward-compatible wrapper scripts that redirect to the new utility
   - Added deprecation warnings to alert users to update their scripts

3. **Symlink for Convenience:**
   - Added a symlink from `scripts/dataset_utils.py` to `data/dataset_utils.py` for convenience

## New Usage

The new unified script provides a command-line interface with subcommands:

```bash
# Download datasets
python data/dataset_utils.py download --dataset coco --data-dir data/coco --year 2017
python data/dataset_utils.py download --dataset coco128 --data-dir data/coco

# View dataset statistics
python data/dataset_utils.py stats --data-dir data/coco

# Create dataset subsets
python data/dataset_utils.py subset --data-dir data/coco --output-dir data/coco_subset --num-images 1000
```

## Function List

The new unified module provides these key functions:

- `download_file()`: Download a file with progress display
- `extract_zip()`: Extract a zip file with progress display
- `download_coco_dataset()`: Download and prepare COCO dataset (full or mini)
- `organize_coco128_dataset()`: Organize COCO128 dataset into standard format
- `check_pycocotools()`: Check and install pycocotools
- `print_dataset_statistics()`: Print statistics about a dataset
- `print_dataset_structure()`: Print directory structure of a dataset
- `create_subset_dataset()`: Create a smaller subset of a dataset
- `create_coco_subset()`: Create a subset of COCO annotations and images

## Backward Compatibility

For backward compatibility, the old script paths still work but will display deprecation warnings:

```bash
# Old scripts - still work but show deprecation warnings
python scripts/download_coco.py --data-dir data/coco --year 2017
python data/download_dataset.py --data-dir data
```

## Future Improvements

Future improvements to the dataset utilities may include:

1. Support for additional datasets (PASCAL VOC, Open Images, etc.)
2. Tools for dataset conversion (between different annotation formats)
3. Dataset visualization and exploration tools
4. Integration with data augmentation pipelines 