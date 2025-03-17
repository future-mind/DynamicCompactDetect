"""
General utilities for YOLOv11 training and inference
"""

import math
import os
import glob
import random
import re
import time
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def set_logging(name=None, verbose=True):
    """
    Sets up logging for the given name
    
    Args:
        name: name of the logger
        verbose: log if verbose
        
    Returns:
        Logger instance
    """
    # Set up logging format
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    formatter = logging.Formatter("%(message)s")
    
    # Set up handlers
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    # Set up logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger


def xywh2xyxy(x):
    """
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    
    Args:
        x: boxes in [x, y, w, h] format
        
    Returns:
        boxes in [x1, y1, x2, y2] format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    """
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]
    
    Args:
        x: boxes in [x1, y1, x2, y2] format
        
    Returns:
        boxes in [x, y, w, h] format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def clip_boxes(boxes, shape):
    """
    Clip boxes to image boundaries
    
    Args:
        boxes: boxes in [x1, y1, x2, y2] format
        shape: image shape (height, width)
        
    Returns:
        clipped boxes
    """
    # Clip bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array
        boxes[:, 0] = np.clip(boxes[:, 0], 0, shape[1])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, shape[0])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, shape[1])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, shape[0])
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Rescale boxes from img1_shape to img0_shape
    
    Args:
        img1_shape: shape of resized image (h, w)
        boxes: boxes in [x1, y1, x2, y2] format
        img0_shape: shape of original image (h, w)
        ratio_pad: ratio of resized image to original image
        
    Returns:
        scaled boxes
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    boxes = clip_boxes(boxes, img0_shape)
    return boxes


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """
    Runs Non-Maximum Suppression (NMS) on inference results
    
    Args:
        prediction: model predictions tensor, shape (batch_size, num_boxes, num_classes + 5)
        conf_thres: confidence threshold
        iou_thres: IoU threshold for NMS
        classes: filter by class, i.e. = [0, 15, 16] for COCO people, cat, dog
        agnostic: class-agnostic NMS
        multi_label: multiple labels per box (adds 0.5ms/img)
        max_det: maximum number of detections per image
        
    Returns:
        list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    
    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        
        # If none remain process next image
        if not x.shape[0]:
            continue
            
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        output[xi] = x[i]
    
    return output


def check_img_size(img_size, s=32):
    """
    Make sure image size is a multiple of stride s
    
    Args:
        img_size: (int or list): image size
        s: stride
        
    Returns:
        new_size: adjusted image size
    """
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), int(s))
        if new_size != img_size:
            logger.warning(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size
    else:  # list i.e. img_size=[640, 480]
        return [max(make_divisible(x, int(s)), int(s)) for x in img_size]


def make_divisible(x, divisor):
    """
    Returns x evenly divisible by divisor
    
    Args:
        x: input value
        divisor: divisor
        
    Returns:
        value divisible by divisor
    """
    return int(np.ceil(x / divisor) * divisor)


def colorstr(*input):
    """
    Colors a string for terminal output
    
    Args:
        *input: input strings
        
    Returns:
        colored string
    """
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args if x in colors) + f'{string}' + colors['end']


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    
    Args:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
        
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - 
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    
    # IoU = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints
    
    Args:
        img: input image
        new_shape: target shape
        color: color for padding
        auto: minimum rectangle
        scaleFill: stretch
        scaleup: scale up
        stride: stride
        
    Returns:
        resized and padded image, ratio, padding
    """
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
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    Increment file or directory path
    i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    
    Args:
        path: path to increment
        exist_ok: existing project/name ok, do not increment
        sep: separator between path and increment
        mkdir: create directory
        
    Returns:
        Incremented path
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(file_path):
    """
    Load YAML configuration file
    
    Args:
        file_path: path to YAML file
        
    Returns:
        config dictionary
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(file_path, data):
    """
    Save dictionary to YAML configuration file
    
    Args:
        file_path: path to YAML file
        data: dictionary to save
    """
    with open(file_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)


def print_args(args=None, show_file=True):
    """
    Print script arguments
    
    Args:
        args: argparse.Namespace
        show_file: show file name in output
    """
    if args is None:
        import sys
        from pathlib import Path
        
        FILE = Path(sys.argv[0]).resolve()
        if show_file and FILE.exists():
            print(f'File: {FILE}')
        args = sys.argv[1:]
        if not args:
            print('No arguments specified.')
            return
        
        for arg in args:
            print(f'Argument: {arg}')
    else:
        from pathlib import Path
        
        FILE = Path(sys.argv[0]).resolve()
        if show_file and FILE.exists():
            print(f'File: {FILE}')
        
        for k, v in vars(args).items():
            print(f'{k}: {v}')
            

def is_ascii(s):
    """
    Check if string contains only ASCII characters
    
    Args:
        s: string to check
        
    Returns:
        True if string contains only ASCII characters
    """
    return len(s.encode()) == len(s)


def coco80_to_coco91_class():
    """
    Converts COCO80 class indices to COCO91 class indices
    
    Returns:
        list of COCO91 class indices
    """
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y 