# Data Directory

This directory is for storing small sample data files for testing and demonstration purposes.

## Dataset Structure

For full datasets, please use the `datasets` directory and follow the instructions in the main README.

## Sample Data

The `samples` directory contains small example images used for demonstrations and tests.

## Data Format

For training and inference, the following data formats are supported:

1. **Images**: JPG, PNG, BMP
2. **Videos**: MP4, AVI
3. **Annotations**: COCO JSON format

## Data Organization

When adding datasets, please follow this organization:

```
datasets/
├── coco/
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── train2017/
│   │   └── [images]
│   └── val2017/
│       └── [images]
└── custom_dataset/
    ├── annotations/
    │   └── instances.json
    ├── train/
    │   └── [images]
    └── val/
        └── [images]
```

## Adding New Datasets

1. Create a subdirectory in `datasets/` for your dataset
2. Ensure images and annotations follow the expected format
3. Use the `scripts/download_coco.py` script as a reference for downloading public datasets

## Note

Large datasets should NOT be committed to the repository. They will be automatically excluded by the `.gitignore` file. 