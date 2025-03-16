#!/usr/bin/env python
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.model import DynamicCompactDetect

def load_model(weights, cfg, device):
    """Load the model"""
    model = DynamicCompactDetect(cfg=cfg)
    ckpt = torch.load(weights, map_location=device)
    # ... existing code ... 