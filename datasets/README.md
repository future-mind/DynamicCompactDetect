# Datasets Directory

This directory is used to store the datasets used for training, validation, and testing. The datasets are NOT included in the repository due to their size.

## Downloading Datasets

### COCO Dataset

To download the COCO dataset, use the provided script:

```bash
python scripts/download_coco.py
```

This will download and extract the COCO dataset to this directory.

### Expected Structure

After downloading, the COCO dataset should have the following structure:

```
datasets/
└── coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017/
    │   └── [images]
    └── val2017/
        └── [images]
```

## Custom Datasets

To use custom datasets:

1. Create a new subdirectory for your dataset
2. Organize images and annotations in the COCO format
3. Update the data loading configuration in `configs/train_configs/custom.yaml`

## Dataset Statistics

| Dataset | Split | Images | Annotations | Classes |
|---------|-------|--------|------------|---------|
| COCO    | train | 118K   | 850K       | 80      |
| COCO    | val   | 5K     | 36K        | 80      |

## Preprocessing

The datasets are automatically preprocessed during training and inference:
- Images are resized to the specified size (default: 640x640)
- Augmentations are applied during training (configurable)
- Pixel values are normalized to [0,1] 