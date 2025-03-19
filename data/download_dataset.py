import os
import argparse
import subprocess
import zipfile
import shutil
from tqdm import tqdm
import requests
import json

COCO_TRAIN_IMAGES = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_IMAGES = "http://images.cocodataset.org/zips/val2017.zip"
COCO_TEST_IMAGES = "http://images.cocodataset.org/zips/test2017.zip"
COCO_TRAIN_VAL_ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def download_file(url, filepath):
    """Download a file from a URL to a local filepath with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(filepath, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(filepath)}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_dir):
    """Extract a zip file to the specified directory."""
    print(f"Extracting {zip_path} to {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting'):
            zip_ref.extract(member, extract_dir)

def prepare_coco_dataset(data_dir, download_dir, splits=None):
    """Download and prepare the COCO dataset for object detection."""
    if splits is None:
        splits = ['train2017', 'val2017']
    
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    coco_dir = os.path.join(data_dir, 'coco')
    os.makedirs(coco_dir, exist_ok=True)
    
    # Download and extract training images
    if 'train2017' in splits:
        train_zip = os.path.join(download_dir, 'train2017.zip')
        if not os.path.exists(train_zip):
            download_file(COCO_TRAIN_IMAGES, train_zip)
        
        os.makedirs(os.path.join(coco_dir, 'train2017'), exist_ok=True)
        if not os.path.exists(os.path.join(coco_dir, 'train2017', '000000000001.jpg')):
            extract_zip(train_zip, coco_dir)
    
    # Download and extract validation images
    if 'val2017' in splits:
        val_zip = os.path.join(download_dir, 'val2017.zip')
        if not os.path.exists(val_zip):
            download_file(COCO_VAL_IMAGES, val_zip)
        
        os.makedirs(os.path.join(coco_dir, 'val2017'), exist_ok=True)
        if not os.path.exists(os.path.join(coco_dir, 'val2017', '000000000139.jpg')):
            extract_zip(val_zip, coco_dir)
    
    # Download and extract test images
    if 'test2017' in splits:
        test_zip = os.path.join(download_dir, 'test2017.zip')
        if not os.path.exists(test_zip):
            download_file(COCO_TEST_IMAGES, test_zip)
        
        os.makedirs(os.path.join(coco_dir, 'test2017'), exist_ok=True)
        if not os.path.exists(os.path.join(coco_dir, 'test2017', '000000000001.jpg')):
            extract_zip(test_zip, coco_dir)
    
    # Download and extract annotations
    annotations_zip = os.path.join(download_dir, 'annotations_trainval2017.zip')
    if not os.path.exists(annotations_zip):
        download_file(COCO_TRAIN_VAL_ANNOTATIONS, annotations_zip)
    
    if not os.path.exists(os.path.join(coco_dir, 'annotations')):
        extract_zip(annotations_zip, coco_dir)
    
    print(f"COCO dataset successfully prepared at {coco_dir}")
    print("Dataset structure:")
    for root, dirs, files in os.walk(coco_dir):
        level = root.replace(coco_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for d in dirs:
            print(f"{sub_indent}{d}/")
        if level <= 1:
            for f in files[:5]:
                print(f"{sub_indent}{f}")
            if len(files) > 5:
                print(f"{sub_indent}... ({len(files)} files in total)")
    
    # Create additional directories for model outputs
    os.makedirs(os.path.join(data_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'results'), exist_ok=True)
    
    return coco_dir

def check_pycocotools():
    """Check and install pycocotools if not already installed."""
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

def print_dataset_stats(coco_dir):
    """Print some basic statistics about the COCO dataset."""
    try:
        from pycocotools.coco import COCO
        
        # Load training annotations
        ann_file = os.path.join(coco_dir, 'annotations', 'instances_train2017.json')
        if os.path.exists(ann_file):
            train_coco = COCO(ann_file)
            
            # Get category information
            cats = train_coco.loadCats(train_coco.getCatIds())
            cat_names = [cat['name'] for cat in cats]
            
            # Get image and annotation counts
            img_ids = train_coco.getImgIds()
            ann_ids = train_coco.getAnnIds()
            
            print("\nCOCO Training Dataset Statistics:")
            print(f"Number of training images: {len(img_ids)}")
            print(f"Number of annotations: {len(ann_ids)}")
            print(f"Number of categories: {len(cats)}")
            print(f"Categories: {', '.join(cat_names[:10])}...")
        
        # Load validation annotations
        ann_file = os.path.join(coco_dir, 'annotations', 'instances_val2017.json')
        if os.path.exists(ann_file):
            val_coco = COCO(ann_file)
            
            # Get image and annotation counts
            img_ids = val_coco.getImgIds()
            ann_ids = val_coco.getAnnIds()
            
            print("\nCOCO Validation Dataset Statistics:")
            print(f"Number of validation images: {len(img_ids)}")
            print(f"Number of annotations: {len(ann_ids)}")
    
    except ImportError:
        print("pycocotools not available. Install it to view dataset statistics.")
    except Exception as e:
        print(f"Error generating dataset statistics: {e}")

def create_coco_subsets(coco_dir, output_dir, num_images=1000):
    """Create smaller subsets of the COCO dataset for quick development and testing."""
    try:
        from pycocotools.coco import COCO
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process training set
        ann_file = os.path.join(coco_dir, 'annotations', 'instances_train2017.json')
        out_ann_file = os.path.join(output_dir, 'instances_train_mini.json')
        out_img_dir = os.path.join(output_dir, 'train_mini')
        os.makedirs(out_img_dir, exist_ok=True)
        
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()[:num_images]  # Take the first N images
        
        # Create annotation subset
        print("Creating training mini dataset...")
        create_subset_annotations(coco, img_ids, out_ann_file)
        
        # Copy selected images
        copy_selected_images(img_ids, coco, os.path.join(coco_dir, 'train2017'), out_img_dir)
        
        # Process validation set (using a smaller number)
        ann_file = os.path.join(coco_dir, 'annotations', 'instances_val2017.json')
        out_ann_file = os.path.join(output_dir, 'instances_val_mini.json')
        out_img_dir = os.path.join(output_dir, 'val_mini')
        os.makedirs(out_img_dir, exist_ok=True)
        
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()[:num_images//5]  # Take fewer validation images
        
        # Create annotation subset
        print("Creating validation mini dataset...")
        create_subset_annotations(coco, img_ids, out_ann_file)
        
        # Copy selected images
        copy_selected_images(img_ids, coco, os.path.join(coco_dir, 'val2017'), out_img_dir)
        
        print(f"Mini datasets created at {output_dir}")
    
    except ImportError:
        print("pycocotools not available. Install it to create dataset subsets.")
    except Exception as e:
        print(f"Error creating dataset subsets: {e}")

def create_subset_annotations(coco, img_ids, output_file):
    """Create a new annotation file with only the selected images."""
    # Load original annotations
    with open(coco.annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Filter images
    images = [img for img in coco_data['images'] if img['id'] in img_ids]
    image_ids = [img['id'] for img in images]
    
    # Filter annotations
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
    
    # Create new data structure
    new_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'images': images,
        'annotations': annotations,
        'categories': coco_data['categories']
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(new_data, f)

def copy_selected_images(img_ids, coco, src_dir, dst_dir):
    """Copy the selected images to the destination directory."""
    for img_id in tqdm(img_ids, desc="Copying images"):
        img_info = coco.loadImgs(img_id)[0]
        filename = img_info['file_name']
        shutil.copy2(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))

def main():
    parser = argparse.ArgumentParser(description='Download and prepare the COCO dataset')
    parser.add_argument('--data-dir', default='../data', help='Directory to store the dataset')
    parser.add_argument('--download-dir', default='../downloads', help='Directory to store the downloaded files')
    parser.add_argument('--splits', nargs='+', default=['train2017', 'val2017'], 
                        choices=['train2017', 'val2017', 'test2017'], 
                        help='Dataset splits to download')
    parser.add_argument('--create-mini', action='store_true', help='Create mini versions of the dataset for quick testing')
    parser.add_argument('--mini-size', type=int, default=1000, help='Number of images in the mini dataset')
    args = parser.parse_args()
    
    # Check and install pycocotools
    check_pycocotools()
    
    # Prepare COCO dataset
    coco_dir = prepare_coco_dataset(args.data_dir, args.download_dir, args.splits)
    
    # Print dataset statistics
    print_dataset_stats(coco_dir)
    
    # Create mini datasets if requested
    if args.create_mini:
        mini_dir = os.path.join(args.data_dir, 'coco_mini')
        create_coco_subsets(coco_dir, mini_dir, args.mini_size)

if __name__ == "__main__":
    main() 