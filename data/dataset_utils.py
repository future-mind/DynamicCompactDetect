#!/usr/bin/env python
"""
Dataset utilities for DynamicCompactDetect.

This module provides functions for downloading and preparing datasets for training,
validation, and testing. It currently supports:
- COCO dataset (full and mini versions)
- COCO128 (lightweight subset for testing)

Future support may include:
- PASCAL VOC
- Open Images
- Custom datasets
"""

import os
import sys
import argparse
import subprocess
import zipfile
import shutil
import json
from pathlib import Path
from tqdm import tqdm
import requests

# URLs for COCO dataset
COCO_URLS = {
    '2017': {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'test_images': 'http://images.cocodataset.org/zips/test2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    },
    '2014': {
        'train_images': 'http://images.cocodataset.org/zips/train2014.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2014.zip',
        'test_images': 'http://images.cocodataset.org/zips/test2014.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    }
}

# URL for COCO128 mini dataset
COCO128_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip'

def download_file(url, filepath, force=False):
    """
    Download a file from a URL to a local filepath with progress bar.
    
    Args:
        url: URL to download
        filepath: Local path to save the file
        force: If True, download even if file exists
    """
    if os.path.exists(filepath) and not force:
        print(f"File {filepath} already exists. Use --force to redownload.")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Try different download methods
    try:
        # Method 1: Use requests with progress bar (preferred)
        print(f"Downloading {url} to {filepath}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB
        
        with open(filepath, 'wb') as f, tqdm(
                desc=f"Downloading {os.path.basename(filepath)}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
    except Exception as e:
        print(f"Error using requests: {e}, trying alternative methods...")
        
        # Method 2: Try using wget if available
        wget_available = shutil.which('wget') is not None
        curl_available = shutil.which('curl') is not None
        
        if wget_available:
            subprocess.run(['wget', '-O', filepath, url], check=True)
        elif curl_available:
            subprocess.run(['curl', '-L', '-o', filepath, url], check=True)
        else:
            print("Error: Neither wget, curl nor python requests worked for downloading.")
            sys.exit(1)
    
    print(f"Download completed: {filepath}")

def extract_zip(zip_path, extract_to, show_progress=True):
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
        show_progress: Whether to show progress bar
    """
    print(f"Extracting {zip_path} to {extract_to}")
    
    # Create directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    # Check if the file exists
    if not os.path.exists(zip_path):
        print(f"Error: File {zip_path} does not exist.")
        return
    
    # Extract the zip file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if show_progress:
                members = zip_ref.infolist()
                for member in tqdm(members, desc=f"Extracting {os.path.basename(zip_path)}"):
                    zip_ref.extract(member, extract_to)
            else:
                zip_ref.extractall(extract_to)
        print(f"Extraction completed: {extract_to}")
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")

def download_coco_dataset(data_dir, year='2017', download_only=False, extract_only=False, mini=False, force=False):
    """
    Download and prepare COCO dataset.
    
    Args:
        data_dir: Base directory to save dataset
        year: COCO dataset year ('2017' or '2014')
        download_only: If True, only download without extracting
        extract_only: If True, only extract without downloading
        mini: If True, download mini COCO dataset (COCO128)
        force: If True, force download even if files exist
    """
    data_dir = Path(data_dir)
    
    # Choose URLs based on mini or full dataset
    if mini:
        print("Preparing mini COCO dataset (COCO128) for testing")
        urls = {
            'mini': COCO128_URL,
        }
        save_paths = {
            'mini': data_dir / 'coco128.zip',
        }
    else:
        print(f"Preparing full COCO {year} dataset")
        if year not in COCO_URLS:
            print(f"Error: COCO {year} dataset URLs not defined")
            return
            
        urls = COCO_URLS[year]
        save_paths = {
            'train_images': data_dir / f'train{year}.zip',
            'val_images': data_dir / f'val{year}.zip',
            'test_images': data_dir / f'test{year}.zip',
            'annotations': data_dir / f'annotations_trainval{year}.zip',
        }
    
    # Create directories
    images_dir = data_dir / 'images'
    annotations_dir = data_dir / 'annotations'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Download files if not extract_only
    if not extract_only:
        if mini:
            download_file(urls['mini'], save_paths['mini'], force)
        else:
            download_file(urls['train_images'], save_paths['train_images'], force)
            download_file(urls['val_images'], save_paths['val_images'], force)
            download_file(urls['annotations'], save_paths['annotations'], force)
            # Test images are optional
            if 'test_images' in urls and urls['test_images']:
                download_file(urls['test_images'], save_paths['test_images'], force)
    
    # Extract files if not download_only
    if not download_only:
        if mini:
            # Extract mini COCO
            extract_zip(save_paths['mini'], data_dir)
            
            # Organize mini dataset
            organize_coco128_dataset(data_dir, images_dir, annotations_dir)
        else:
            # Extract full COCO dataset
            extract_zip(save_paths['train_images'], images_dir)
            extract_zip(save_paths['val_images'], images_dir)
            extract_zip(save_paths['annotations'], data_dir)
            
            # Extract test images if available
            if 'test_images' in save_paths and os.path.exists(save_paths['test_images']):
                extract_zip(save_paths['test_images'], images_dir)
    
    print(f"COCO dataset preparation completed in {data_dir}")
    print_dataset_structure(data_dir)

def organize_coco128_dataset(data_dir, images_dir, annotations_dir):
    """
    Organize COCO128 dataset into the standard COCO format.
    
    Args:
        data_dir: Base directory containing coco128 folder
        images_dir: Directory to store images
        annotations_dir: Directory to store annotations
    """
    # Paths in the coco128 structure
    coco128_images = data_dir / 'coco128' / 'images' / 'train2017'
    coco128_labels = data_dir / 'coco128' / 'labels' / 'train2017'
    
    if not coco128_images.exists() or not coco128_labels.exists():
        print("Error: COCO128 dataset structure not found")
        return
    
    # Create target directories
    train_images_dir = images_dir / 'train2017'
    val_images_dir = images_dir / 'val2017'
    train_labels_dir = annotations_dir / 'train2017'
    val_labels_dir = annotations_dir / 'val2017'
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Copy images
    print("Organizing COCO128 images...")
    for img_path in tqdm(list(coco128_images.glob('*.jpg')), desc="Copying images"):
        img_id = int(img_path.stem)
        
        # Copy to train directory
        shutil.copy(img_path, train_images_dir / img_path.name)
        
        # Copy first 20 images to validation directory as well
        if img_id < 20:
            shutil.copy(img_path, val_images_dir / img_path.name)
    
    # Copy labels (annotations)
    print("Organizing COCO128 annotations...")
    for label_path in tqdm(list(coco128_labels.glob('*.txt')), desc="Copying annotations"):
        img_id = int(label_path.stem)
        
        # Copy to train directory
        shutil.copy(label_path, train_labels_dir / label_path.name)
        
        # Copy first 20 annotations to validation directory as well
        if img_id < 20:
            shutil.copy(label_path, val_labels_dir / label_path.name)
    
    # Convert YOLO format to COCO format if needed
    # This step would require additional functionality to convert the annotation format
    print("Note: YOLO format annotations need to be converted to COCO format for full compatibility")

def check_pycocotools():
    """
    Check and install pycocotools if not already installed.
    """
    try:
        import pycocotools
        print("pycocotools already installed")
    except ImportError:
        print("Installing pycocotools...")
        try:
            from platform import system
            if system() == "Windows":
                # Windows requires Microsoft Visual C++ Build Tools
                subprocess.check_call(['pip', 'install', 'pycocotools-windows'])
            else:
                # Linux or MacOS
                subprocess.check_call(['pip', 'install', 'pycocotools'])
            print("pycocotools successfully installed")
        except Exception as e:
            print(f"Failed to install pycocotools: {e}")
            print("Please install it manually following the instructions at: https://github.com/cocodataset/cocoapi")

def print_dataset_statistics(data_dir, year='2017'):
    """
    Print statistics about the COCO dataset.
    
    Args:
        data_dir: Base directory containing the dataset
        year: COCO dataset year
    """
    try:
        from pycocotools.coco import COCO
        
        data_dir = Path(data_dir)
        
        # Load training annotations
        train_ann_file = data_dir / 'annotations' / f'instances_train{year}.json'
        if train_ann_file.exists():
            train_coco = COCO(str(train_ann_file))
            
            # Get category information
            cats = train_coco.loadCats(train_coco.getCatIds())
            cat_names = [cat['name'] for cat in cats]
            
            # Get image and annotation counts
            train_img_ids = train_coco.getImgIds()
            train_ann_ids = train_coco.getAnnIds()
            
            print("\nCOCO Training Dataset Statistics:")
            print(f"Number of training images: {len(train_img_ids)}")
            print(f"Number of annotations: {len(train_ann_ids)}")
            print(f"Number of categories: {len(cats)}")
            print(f"Categories: {', '.join(cat_names[:10])}...")
        
        # Load validation annotations
        val_ann_file = data_dir / 'annotations' / f'instances_val{year}.json'
        if val_ann_file.exists():
            val_coco = COCO(str(val_ann_file))
            
            # Get image and annotation counts
            val_img_ids = val_coco.getImgIds()
            val_ann_ids = val_coco.getAnnIds()
            
            print("\nCOCO Validation Dataset Statistics:")
            print(f"Number of validation images: {len(val_img_ids)}")
            print(f"Number of annotations: {len(val_ann_ids)}")
    
    except ImportError:
        print("pycocotools not available. Install it to view dataset statistics.")
    except Exception as e:
        print(f"Error generating dataset statistics: {e}")

def print_dataset_structure(data_dir):
    """
    Print the structure of the dataset directory.
    
    Args:
        data_dir: Base directory containing the dataset
    """
    data_dir = Path(data_dir)
    print("\nDataset structure:")
    
    # Maximum depth to display
    max_depth = 3
    
    # Maximum files to display per directory
    max_files = 5
    
    for root, dirs, files in os.walk(data_dir):
        root_path = Path(root)
        level = len(root_path.relative_to(data_dir).parts)
        
        if level > max_depth:
            continue
            
        indent = ' ' * 4 * level
        print(f"{indent}{root_path.name}/")
        
        # Print directories at this level
        for d in sorted(dirs):
            if not d.startswith('.'):  # Skip hidden directories
                sub_indent = ' ' * 4 * (level + 1)
                print(f"{sub_indent}{d}/")
        
        # Print files at this level (limited number)
        shown_files = [f for f in sorted(files) if not f.startswith('.')][:max_files]
        for f in shown_files:
            sub_indent = ' ' * 4 * (level + 1)
            print(f"{sub_indent}{f}")
        
        # Show count if there are more files
        if len(files) > max_files:
            sub_indent = ' ' * 4 * (level + 1)
            print(f"{sub_indent}... ({len(files)} files in total)")

def create_subset_dataset(data_dir, output_dir, num_images=1000, year='2017'):
    """
    Create a smaller subset of the COCO dataset for development and testing.
    
    Args:
        data_dir: Base directory containing the full dataset
        output_dir: Directory to save the subset
        num_images: Number of images to include in the subset
        year: COCO dataset year
    """
    try:
        from pycocotools.coco import COCO
        
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process training set
        train_ann_file = data_dir / 'annotations' / f'instances_train{year}.json'
        train_img_dir = data_dir / 'images' / f'train{year}'
        
        out_train_ann_file = output_dir / f'instances_train_subset.json'
        out_train_img_dir = output_dir / 'train_subset'
        os.makedirs(out_train_img_dir, exist_ok=True)
        
        # Check if files exist
        if not train_ann_file.exists() or not train_img_dir.exists():
            print(f"Error: Training data not found at {train_ann_file} or {train_img_dir}")
            return
        
        # Create training subset
        print("Creating training subset dataset...")
        create_coco_subset(train_ann_file, train_img_dir, 
                          out_train_ann_file, out_train_img_dir, 
                          num_images)
        
        # Process validation set (using fewer images)
        val_ann_file = data_dir / 'annotations' / f'instances_val{year}.json'
        val_img_dir = data_dir / 'images' / f'val{year}'
        
        out_val_ann_file = output_dir / f'instances_val_subset.json'
        out_val_img_dir = output_dir / 'val_subset'
        os.makedirs(out_val_img_dir, exist_ok=True)
        
        # Check if files exist
        if not val_ann_file.exists() or not val_img_dir.exists():
            print(f"Error: Validation data not found at {val_ann_file} or {val_img_dir}")
            return
            
        # Create validation subset
        print("Creating validation subset dataset...")
        create_coco_subset(val_ann_file, val_img_dir, 
                          out_val_ann_file, out_val_img_dir, 
                          num_images // 5)  # Fewer validation images
        
        print(f"Subset dataset created at {output_dir}")
        
    except ImportError:
        print("pycocotools not available. Install it to create dataset subsets.")
    except Exception as e:
        print(f"Error creating dataset subset: {e}")

def create_coco_subset(ann_file, img_dir, out_ann_file, out_img_dir, num_images):
    """
    Create a subset of COCO annotations and copy corresponding images.
    
    Args:
        ann_file: Path to the annotation file
        img_dir: Directory containing the images
        out_ann_file: Path to save the subset annotation file
        out_img_dir: Directory to save the subset images
        num_images: Number of images to include
    """
    try:
        from pycocotools.coco import COCO
        
        # Load COCO annotations
        coco = COCO(str(ann_file))
        
        # Get image IDs (randomly sample to get diverse images)
        import random
        img_ids = coco.getImgIds()
        random.seed(42)  # For reproducibility
        selected_img_ids = random.sample(img_ids, min(num_images, len(img_ids)))
        
        # Create new annotation file
        dataset = {
            'info': coco.dataset.get('info', {}),
            'licenses': coco.dataset.get('licenses', []),
            'categories': coco.dataset.get('categories', []),
            'images': [],
            'annotations': []
        }
        
        # Add selected images and their annotations
        for img_id in tqdm(selected_img_ids, desc="Processing annotations"):
            # Add image info
            img_info = coco.loadImgs(img_id)[0]
            dataset['images'].append(img_info)
            
            # Add annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            dataset['annotations'].extend(anns)
            
            # Copy image file
            src_file = Path(img_dir) / img_info['file_name']
            if src_file.exists():
                dst_file = Path(out_img_dir) / img_info['file_name']
                shutil.copy(src_file, dst_file)
        
        # Save the new annotation file
        with open(out_ann_file, 'w') as f:
            json.dump(dataset, f)
            
        print(f"Created subset with {len(dataset['images'])} images and {len(dataset['annotations'])} annotations")
        
    except Exception as e:
        print(f"Error creating COCO subset: {e}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Dataset utilities for DynamicCompactDetect')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Download COCO dataset command
    download_parser = subparsers.add_parser('download', help='Download and prepare dataset')
    download_parser.add_argument('--dataset', type=str, default='coco', 
                               choices=['coco', 'coco128'], 
                               help='Dataset to download')
    download_parser.add_argument('--data-dir', type=str, default='data', 
                               help='Directory to save dataset')
    download_parser.add_argument('--year', type=str, default='2017',
                               choices=['2014', '2017'], 
                               help='COCO dataset year (only for COCO)')
    download_parser.add_argument('--download-only', action='store_true',
                               help='Only download without extracting')
    download_parser.add_argument('--extract-only', action='store_true',
                               help='Only extract without downloading')
    download_parser.add_argument('--force', action='store_true',
                               help='Force download even if files exist')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Print dataset statistics')
    stats_parser.add_argument('--data-dir', type=str, default='data/coco', 
                            help='Dataset directory')
    stats_parser.add_argument('--year', type=str, default='2017',
                            choices=['2014', '2017'], 
                            help='COCO dataset year')
    
    # Create subset command
    subset_parser = subparsers.add_parser('subset', help='Create dataset subset')
    subset_parser.add_argument('--data-dir', type=str, default='data/coco', 
                             help='Full dataset directory')
    subset_parser.add_argument('--output-dir', type=str, default='data/coco_subset', 
                             help='Output directory for subset')
    subset_parser.add_argument('--num-images', type=int, default=1000, 
                             help='Number of images in subset')
    subset_parser.add_argument('--year', type=str, default='2017',
                             choices=['2014', '2017'], 
                             help='COCO dataset year')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        if args.dataset == 'coco':
            download_coco_dataset(
                args.data_dir, 
                args.year, 
                args.download_only, 
                args.extract_only,
                mini=False,
                force=args.force
            )
        elif args.dataset == 'coco128':
            download_coco_dataset(
                args.data_dir, 
                args.year, 
                args.download_only, 
                args.extract_only,
                mini=True,
                force=args.force
            )
    
    elif args.command == 'stats':
        check_pycocotools()
        print_dataset_statistics(args.data_dir, args.year)
    
    elif args.command == 'subset':
        check_pycocotools()
        create_subset_dataset(args.data_dir, args.output_dir, args.num_images, args.year)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 