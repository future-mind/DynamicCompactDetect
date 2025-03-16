import argparse
import torch
import cv2
import numpy as np
import yaml
import os
from pathlib import Path
from models import DynamicCompactDetect, load_model
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.plots import Colors, plot_one_box


def parse_args():
    parser = argparse.ArgumentParser(description='DynamicCompact-Detect Inference')
    parser.add_argument('--weights', type=str, default='runs/train/exp/best.pt', help='model weights path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--source', type=str, default='datasets/coco/val2017', help='file/dir of images')
    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument('--save-img', action='store_true', help='save output images')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--max-det', type=int, default=100, help='maximum detections per image')
    return parser.parse_args()


def detect(args):
    # Create output directory
    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize device
    device = select_device(args.device)
    
    # Load model
    print(f'Loading model from {args.weights}...')
    if args.weights.endswith('.pt'):  # PyTorch format
        try:
            # Add argparse.Namespace to safe globals list for PyTorch 2.6+
            try:
                import argparse
                from torch.serialization import add_safe_globals
                add_safe_globals([argparse.Namespace])
            except (ImportError, AttributeError):
                pass
            
            # Try loading with weights_only=False first (for compatibility with PyTorch 2.6+)
            try:
                ckpt = torch.load(args.weights, map_location=device, weights_only=False)
            except TypeError:  # Older PyTorch versions don't have weights_only parameter
                ckpt = torch.load(args.weights, map_location=device)
            
            model = DynamicCompactDetect(cfg=ckpt.get('cfg')).to(device)
            model.load_state_dict(ckpt['model'], strict=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Trying to load model directly...")
            model = DynamicCompactDetect(cfg=args.weights).to(device)
    else:
        model = DynamicCompactDetect(cfg=args.weights).to(device)
    
    # Load class names
    with open('data/coco.yaml', 'r') as f:
        data_dict = yaml.safe_load(f)
    names = data_dict['names']
    
    # Set model to evaluation mode
    model.eval()
    
    # Load input source
    source = Path(args.source)
    if source.is_dir():
        files = sorted(source.glob('**/*.jpg')) + sorted(source.glob('**/*.png'))
    elif source.is_file() and source.suffix in ('.jpg', '.jpeg', '.png'):
        files = [source]
    else:
        raise Exception(f'Invalid source: {source}')
    
    # Set up colors for visualization
    colors = Colors()
    
    # Process each file
    for file_idx, file in enumerate(files):
        # Read image
        img0 = cv2.imread(str(file))
        if img0 is None:
            print(f'Skipping {file}: Unable to read image')
            continue
        
        # Preprocess image
        img = letterbox(img0, args.img_size)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img.unsqueeze(0)  # add batch dimension
        
        # Inference
        with torch.no_grad():
            pred = model(img)
        
        # Apply NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, max_det=args.max_det)
        
        # Process detections
        for i, det in enumerate(pred):  # per image
            im0 = img0.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    print(f"Image {file_idx}/{len(files)}: {n} {names[int(c)]}")
                
                # Draw bounding boxes
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=2)
            
            # Save results
            if args.save_img:
                save_path = save_dir / file.name
                cv2.imwrite(str(save_path), im0)
                print(f'Saved {save_path}')
        
    print(f'Done. Results saved to {save_dir}')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while maintaining aspect ratio."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def select_device(device=''):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'DynamicCompact-Detect 🚀 '
    device = str(device).strip().lower().replace('cuda:', '')
    cpu = device == 'cpu'
    
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'
    
    cuda = not cpu and torch.cuda.is_available()
    
    if cuda:
        devices = device.split(',') if device else '0'
        n = len(devices)
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"
    else:
        s += 'CPU\n'
    
    print(s)
    return torch.device('cuda:0' if cuda else 'cpu')


if __name__ == '__main__':
    args = parse_args()
    detect(args) 