import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrow, FancyBboxPatch
import os

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 70)
ax.axis('off')

# Colors
colors = {
    'backbone': '#3498db',  # Blue
    'dynamic_block': '#e74c3c',  # Red
    'fpn': '#2ecc71',  # Green
    'detection': '#f39c12',  # Orange
    'arrow': '#7f8c8d',  # Gray
    'bg': '#ecf0f1',  # Light gray
    'text': '#2c3e50',  # Dark blue
    'gate': '#9b59b6'  # Purple
}

# Function to draw a labeled box
def draw_box(x, y, width, height, label, color, alpha=0.8, fontsize=10):
    rect = Rectangle((x, y), width, height, facecolor=color, alpha=alpha, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=fontsize, 
            fontweight='bold', color=colors['text'])
    return rect

# Function to draw an arrow
def draw_arrow(x1, y1, x2, y2, width=0.5):
    arrow = FancyArrow(x1, y1, x2-x1, y2-y1, width=width, length_includes_head=True, 
                      head_width=2*width, head_length=2*width, fc=colors['arrow'], ec='black')
    ax.add_patch(arrow)

# Draw title
ax.text(50, 66, 'DynamicCompactDetect Architecture', ha='center', va='center', 
        fontsize=16, fontweight='bold', color=colors['text'])

# Draw input
draw_box(10, 55, 15, 8, 'Input Image', colors['bg'], 0.7)

# Dynamic Backbone
backbone_x, backbone_y = 5, 20
backbone_w, backbone_h = 25, 30
backbone = draw_box(backbone_x, backbone_y, backbone_w, backbone_h, '', colors['backbone'], 0.3)

# Add backbone components (stages)
stage_height = 5
for i, name in enumerate(['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']):
    y = backbone_y + backbone_h - (i+1)*6
    draw_box(backbone_x + 2, y, backbone_w - 4, stage_height, name, colors['backbone'], 0.7)

# Add dynamic blocks with gating
block_width, block_height = 15, 3
block_x = 35
for i in range(4):
    block_y = 28 - i*8
    
    # Dynamic block
    draw_box(block_x, block_y, block_width, block_height, f'Dynamic Block {i+1}', colors['dynamic_block'], 0.8)
    
    # Gating module
    gate_size = 3
    gate_x = block_x - 5
    gate_y = block_y
    draw_box(gate_x, gate_y, gate_size, gate_size, 'G', colors['gate'], 0.9, fontsize=8)
    
    # Connect from backbone
    stage_end_x = backbone_x + backbone_w
    stage_y = backbone_y + backbone_h - (i+1)*6 + stage_height/2
    draw_arrow(stage_end_x, stage_y, gate_x, gate_y + gate_size/2)
    
    # Connect gate to block
    draw_arrow(gate_x + gate_size, gate_y + gate_size/2, block_x, block_y + block_height/2)
    
    # Show skip connection
    skip_arrow_y = block_y + block_height + 1
    ax.plot([gate_x + gate_size/2, block_x + block_width + 5, block_x + block_width + 5],
            [gate_y + gate_size + 0.5, gate_y + gate_size + 0.5, block_y + block_height/2],
            'k--', alpha=0.7)
    
    # Early exit arrow for some blocks
    if i in [1, 3]:
        draw_arrow(block_x + block_width, block_y, block_x + block_width + 10, block_y - 5)
        draw_box(block_x + block_width + 10, block_y - 8, 10, 6, 'Early Exit', colors['detection'], 0.7, fontsize=8)

# FPN
fpn_x, fpn_y = 55, 20
fpn_w, fpn_h = 15, 30
fpn = draw_box(fpn_x, fpn_y, fpn_w, fpn_h, 'Feature\nPyramid\nNetwork', colors['fpn'], 0.7)

# Connect dynamic blocks to FPN
for i in range(4):
    block_y = 28 - i*8 + block_height/2
    draw_arrow(block_x + block_width, block_y, fpn_x, block_y)

# Detection heads
head_w, head_h = 15, 6
head_x = 75
for i in range(3):
    head_y = 40 - i*10
    draw_box(head_x, head_y, head_w, head_h, f'Detection\nHead {i+1}', colors['detection'], 0.7)
    draw_arrow(fpn_x + fpn_w, head_y + head_h/2, head_x, head_y + head_h/2)

# Output boxes
output_x = 95
for i in range(3):
    output_y = 40 - i*10
    out_w, out_h = 3, 3
    ax.add_patch(Rectangle((output_x, output_y + (head_h - out_h)/2), out_w, out_h, 
                         facecolor='white', edgecolor='black'))
    draw_arrow(head_x + head_w, output_y + head_h/2, output_x, output_y + head_h/2)

# Input arrow
draw_arrow(17.5, 55, 17.5, 50)

# Connect input to backbone
draw_arrow(17.5, 50, 17.5, backbone_y + backbone_h)

# Add legend
legend_x, legend_y = 5, 5
legend_items = [
    ('Backbone', colors['backbone']),
    ('Dynamic Block', colors['dynamic_block']),
    ('Gating Module', colors['gate']),
    ('Feature Pyramid', colors['fpn']),
    ('Detection Head', colors['detection'])
]

for i, (label, color) in enumerate(legend_items):
    rect_x = legend_x + i*19
    rect = Rectangle((rect_x, legend_y), 3, 3, facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(rect_x + 4, legend_y + 1.5, label, va='center', fontsize=8, color=colors['text'])

# Add annotations
ax.annotate('Dynamic computation paths\nbased on input complexity',
           xy=(35, 45), xytext=(25, 50),
           arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
           fontsize=8)

ax.annotate('Early exits allow faster\ninference for simple inputs',
           xy=(65, 23), xytext=(55, 15),
           arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
           fontsize=8)

# Add skip connection annotation
ax.annotate('Skip connections\nbypass computation',
           xy=(40, 35), xytext=(30, 40),
           arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
           fontsize=8)

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dcd_architecture.png'), 
            dpi=300, bbox_inches='tight')
print(f"Generated architecture diagram at {os.path.dirname(os.path.abspath(__file__))}/dcd_architecture.png") 