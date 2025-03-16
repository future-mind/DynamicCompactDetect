#!/usr/bin/env python3
"""
Download and extract COCO dataset.
This script downloads the COCO dataset and extracts it to the datasets directory.
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def parse_args():
    parser = argparse.ArgumentParser(description='Download COCO dataset')
    parser.add_argument('--data-dir', type=str, default='datasets/coco',
                        help='path to save COCO dataset')
    parser.add_argument('--download-val', action='store_true',
                        help='download validation data only')
    parser.add_argument('--download-train', action='store_true',
                        help='download training data')
    parser.add_argument('--download-test', action='store_true',
                        help='download test data')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='do not remove downloaded archives after extraction')
    args = parser.parse_args()
    
    # Default behavior: download validation set only
    if not args.download_val and not args.download_train and not args.download_test:
        args.download_val = True
    
    return args

def download_file(url, save_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {url} to {save_path}")
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=save_path.name) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))

def extract_file(archive_path, extract_dir):
    """Extract a zip or tar file."""
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = Path(archive_path)
    print(f"Extracting {archive_path} to {extract_dir}")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc=f"Extracting {archive_path.name}"):
                zip_ref.extract(member, extract_dir)
    elif archive_path.suffix in ['.tar', '.gz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            for member in tqdm(tar_ref.getmembers(), desc=f"Extracting {archive_path.name}"):
                tar_ref.extract(member, extract_dir)
    else:
        print(f"Unsupported archive format: {archive_path.suffix}")
        return False
    
    return True

def download_coco(args):
    """Download and extract COCO dataset."""
    coco_dir = Path(args.data_dir)
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO dataset URLs
    coco_urls = {
        'val2017': 'http://images.cocodataset.org/zips/val2017.zip',
        'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
        'test2017': 'http://images.cocodataset.org/zips/test2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    }
    
    downloaded_files = []
    
    # Download annotations
    annotations_zip = coco_dir / 'annotations_trainval2017.zip'
    if args.download_val or args.download_train:
        if not (coco_dir / 'annotations').exists():
            download_file(coco_urls['annotations'], annotations_zip)
            extract_file(annotations_zip, coco_dir)
            downloaded_files.append(annotations_zip)
    
    # Download validation data
    if args.download_val:
        val_zip = coco_dir / 'val2017.zip'
        if not (coco_dir / 'val2017').exists():
            download_file(coco_urls['val2017'], val_zip)
            extract_file(val_zip, coco_dir)
            downloaded_files.append(val_zip)
    
    # Download training data
    if args.download_train:
        train_zip = coco_dir / 'train2017.zip'
        if not (coco_dir / 'train2017').exists():
            download_file(coco_urls['train2017'], train_zip)
            extract_file(train_zip, coco_dir)
            downloaded_files.append(train_zip)
    
    # Download test data
    if args.download_test:
        test_zip = coco_dir / 'test2017.zip'
        if not (coco_dir / 'test2017').exists():
            download_file(coco_urls['test2017'], test_zip)
            extract_file(test_zip, coco_dir)
            downloaded_files.append(test_zip)
    
    # Clean up (remove zip files)
    if not args.no_cleanup:
        print("Cleaning up downloaded archives...")
        for file_path in downloaded_files:
            if file_path.exists():
                print(f"Removing {file_path}")
                file_path.unlink()
    
    # Print dataset information
    print("\nCOCO Dataset Information:")
    
    # Check and print directory sizes
    dirs_to_check = ['annotations', 'val2017', 'train2017', 'test2017']
    for dir_name in dirs_to_check:
        dir_path = coco_dir / dir_name
        if dir_path.exists():
            # Count files
            file_count = len(list(dir_path.glob('*')))
            # Get directory size
            dir_size_bytes = sum(f.stat().st_size for f in dir_path.glob('**/*') if f.is_file())
            dir_size_mb = dir_size_bytes / (1024 * 1024)
            print(f"  {dir_name}: {file_count} files, {dir_size_mb:.1f} MB")
        else:
            print(f"  {dir_name}: Not downloaded")
    
    print("\nDataset download complete.")
    print(f"COCO dataset saved to {coco_dir.resolve()}")

def main():
    args = parse_args()
    download_coco(args)

if __name__ == "__main__":
    main() 