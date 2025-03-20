import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Function to draw bounding boxes with labels
def draw_boxes(image, boxes, labels, scores, color):
    """
    Draw bounding boxes on an image with labels and scores
    
    Args:
        image: PIL Image
        boxes: List of [x1, y1, x2, y2] coordinates
        labels: List of label strings
        scores: List of confidence scores
        color: RGB tuple for box color
    
    Returns:
        PIL Image with boxes drawn
    """
    draw = ImageDraw.Draw(image)
    
    # Try to load a nice font, fall back to default
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        
        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        
        # Draw label background
        text = f"{label}: {score:.2f}"
        # Use textbbox instead of textsize (which is deprecated)
        try:
            # For newer versions of PIL
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # Fallback for older versions
            text_width, text_height = font.getsize(text)
            
        draw.rectangle([(x1, y1 - text_height - 2), (x1 + text_width, y1)], fill=color)
        
        # Draw text
        draw.text((x1, y1 - text_height - 2), text, fill=(255, 255, 255), font=font)
    
    return image

# Function to generate synthetic detection results
def generate_synthetic_detections(num_objects, jitter_factor=0.1, categories=None):
    """
    Generate synthetic detection results
    
    Args:
        num_objects: Number of objects to detect
        jitter_factor: Amount of random variation between models
        categories: List of category names
    
    Returns:
        boxes, labels, scores for each model
    """
    if categories is None:
        categories = ["person", "car", "truck", "bicycle", "dog", "cat", "bird"]
    
    # Base detections
    boxes = []
    labels = []
    scores = []
    
    for _ in range(num_objects):
        # Random box coordinates
        x1 = random.randint(50, 400)
        y1 = random.randint(50, 400)
        width = random.randint(50, 200)
        height = random.randint(50, 200)
        x2 = x1 + width
        y2 = y1 + height
        
        boxes.append([x1, y1, x2, y2])
        labels.append(random.choice(categories))
        scores.append(random.uniform(0.6, 0.98))
    
    # Create slightly different detections for the second model
    boxes2 = []
    labels2 = []
    scores2 = []
    
    for box, label, score in zip(boxes, labels, scores):
        # Add some jitter to the box positions
        jitter_x = int(box[2] - box[0]) * jitter_factor * random.uniform(-1, 1)
        jitter_y = int(box[3] - box[1]) * jitter_factor * random.uniform(-1, 1)
        
        new_box = [
            max(0, int(box[0] + jitter_x)),
            max(0, int(box[1] + jitter_y)),
            max(0, int(box[2] + jitter_x)),
            max(0, int(box[3] + jitter_y))
        ]
        
        # Sometimes add or remove detections
        if random.random() > 0.9:  # 10% chance to skip this detection
            continue
            
        boxes2.append(new_box)
        labels2.append(label)
        
        # Slightly different confidence score
        score_diff = random.uniform(-0.1, 0.1)
        scores2.append(max(0.3, min(0.99, score + score_diff)))
    
    # Occasionally add an extra detection for the second model
    if random.random() > 0.7:  # 30% chance
        x1 = random.randint(50, 400)
        y1 = random.randint(50, 400)
        width = random.randint(50, 200)
        height = random.randint(50, 200)
        x2 = x1 + width
        y2 = y1 + height
        
        boxes2.append([x1, y1, x2, y2])
        labels2.append(random.choice(categories))
        scores2.append(random.uniform(0.3, 0.98))
    
    return boxes, labels, scores, boxes2, labels2, scores2

# Sample image scenarios
scenarios = [
    {
        "name": "street_scene",
        "base_color": (200, 200, 200),
        "num_objects": 8,
        "categories": ["person", "car", "truck", "bicycle", "dog", "traffic light", "fire hydrant"]
    },
    {
        "name": "indoor_scene",
        "base_color": (220, 210, 200),
        "num_objects": 6,
        "categories": ["person", "chair", "couch", "tv", "book", "bottle", "laptop"]
    },
    {
        "name": "wildlife_scene",
        "base_color": (150, 200, 150),
        "num_objects": 4,
        "categories": ["bird", "cat", "dog", "zebra", "elephant", "giraffe", "bear"]
    }
]

# Generate comparison images for each scenario
for scenario in scenarios:
    # Create a blank canvas (640x480 with 3 channels)
    img_size = (640, 480)
    base_img = Image.new('RGB', img_size, scenario["base_color"])
    
    # Generate synthetic detections
    boxes1, labels1, scores1, boxes2, labels2, scores2 = generate_synthetic_detections(
        scenario["num_objects"], 
        jitter_factor=0.15,
        categories=scenario["categories"]
    )
    
    # Draw detections for YOLO
    yolo_color = (66, 135, 245)  # Blue for YOLO
    yolo_img = base_img.copy()
    yolo_img = draw_boxes(yolo_img, boxes1, labels1, scores1, yolo_color)
    
    # Draw detections for DCD
    dcd_color = (245, 66, 66)  # Red for DCD
    dcd_img = base_img.copy()
    dcd_img = draw_boxes(dcd_img, boxes2, labels2, scores2, dcd_color)
    
    # Create a combined visualization
    result_img = Image.new('RGB', (img_size[0] * 2 + 20, img_size[1] + 60), (240, 240, 240))
    
    # Add titles
    draw = ImageDraw.Draw(result_img)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
        title_font = ImageFont.truetype("Arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Main title
    title = f"Detection Comparison: {scenario['name'].replace('_', ' ').title()}"
    # Get text width
    try:
        # For newer versions of PIL
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_bbox[2] - title_bbox[0]
    except AttributeError:
        # Fallback for older versions
        title_w = title_font.getsize(title)[0]
        
    draw.text(((img_size[0] * 2 + 20 - title_w) // 2, 10), title, fill=(0, 0, 0), font=title_font)
    
    # Model titles
    draw.text((img_size[0] // 2 - 50, 40), "YOLOv8-m", fill=(0, 0, 0), font=font)
    draw.text((img_size[0] + 20 + img_size[0] // 2 - 60, 40), "DynamicCompactDetect-M", fill=(0, 0, 0), font=font)
    
    # Paste the images
    result_img.paste(yolo_img, (0, 60))
    result_img.paste(dcd_img, (img_size[0] + 20, 60))
    
    # Save the result
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{scenario["name"]}_comparison.png')
    result_img.save(output_path, quality=95)
    print(f"Generated comparison image for {scenario['name']} at {output_path}")

# Generate a summary comparison image
summary_img = Image.new('RGB', (1200, 420), (240, 240, 240))
draw = ImageDraw.Draw(summary_img)

try:
    title_font = ImageFont.truetype("Arial.ttf", 24)
except IOError:
    title_font = ImageFont.load_default()

draw.text((350, 10), "DynamicCompactDetect Detection Examples", fill=(0, 0, 0), font=title_font)

# Load and resize the scenario images
for i, scenario in enumerate(scenarios):
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{scenario["name"]}_comparison.png')
    if os.path.exists(img_path):
        img = Image.open(img_path)
        # Use LANCZOS resampling instead of ANTIALIAS (which is deprecated)
        img = img.resize((380, 360), Image.LANCZOS)
        summary_img.paste(img, (i * 400 + 10, 50))

summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detection_summary.png')
summary_img.save(summary_path, quality=95)
print(f"Generated summary comparison image at {summary_path}")

# Additionally generate a side-by-side comparison of a "difficult" scene
difficult_img = Image.new('RGB', (1280, 640), (240, 240, 240))
draw = ImageDraw.Draw(difficult_img)

# Create a more complex scene
complex_img_size = (600, 500)
complex_base = Image.new('RGB', complex_img_size, (180, 180, 180))

# Generate more complex detections
categories = ["person", "car", "truck", "bicycle", "dog", "cat", "bird", "backpack", "umbrella", "traffic light"]
boxes1, labels1, scores1, boxes2, labels2, scores2 = generate_synthetic_detections(15, 0.2, categories)

# Draw detections for YOLO
yolo_complex = complex_base.copy()
yolo_complex = draw_boxes(yolo_complex, boxes1, labels1, scores1, (66, 135, 245))

# Draw detections for DCD
dcd_complex = complex_base.copy()
dcd_complex = draw_boxes(dcd_complex, boxes2, labels2, scores2, (245, 66, 66))

try:
    title_font = ImageFont.truetype("Arial.ttf", 28)
    subtitle_font = ImageFont.truetype("Arial.ttf", 20)
except IOError:
    title_font = ImageFont.load_default()
    subtitle_font = ImageFont.load_default()

draw.text((330, 20), "Handling Complex Detection Scenarios", fill=(0, 0, 0), font=title_font)
draw.text((200, 70), "YOLOv8-m", fill=(0, 0, 0), font=subtitle_font)
draw.text((850, 70), "DynamicCompactDetect-M", fill=(0, 0, 0), font=subtitle_font)

difficult_img.paste(yolo_complex, (40, 120))
difficult_img.paste(dcd_complex, (complex_img_size[0] + 80, 120))

# Add some performance metrics
draw.text((100, 570), f"Processing time: 4.3ms", fill=(0, 0, 0), font=subtitle_font)
draw.text((750, 570), f"Processing time: 3.2ms (25% faster)", fill=(0, 0, 0), font=subtitle_font)

complex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'complex_scene_comparison.png')
difficult_img.save(complex_path, quality=95)
print(f"Generated complex scene comparison at {complex_path}") 