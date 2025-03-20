import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Data for plotting
models = ['YOLOv8-n', 'DCD-S', 'YOLOv8-s', 'DCD-M', 'YOLOv8-m', 'DCD-L', 'YOLOv8-l']
map_values = [37.3, 45.6, 44.9, 50.9, 50.2, 53.2, 52.9]  # mAP@0.5:0.95
fps_values = [905, 512, 428, 305, 232, 196, 165]  # FPS

# Calculate relative size compared to maximum for visual impact
max_fps = max(fps_values)
fps_normalized = [fps/max_fps * 50 for fps in fps_values]  # Scale to make bars visible

# Set up the figure with two subplots sharing x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.tight_layout(pad=5.0)

# Define colors for different model families
colors = ['#3498db', '#e74c3c', '#3498db', '#e74c3c', '#3498db', '#e74c3c', '#3498db']
hatches = ['', '///', '', '///', '', '///', '']

# Create the mAP plot
bars1 = ax1.bar(models, map_values, color=colors, width=0.6, edgecolor='black', hatch=hatches)
ax1.set_ylabel('mAP@0.5:0.95', fontsize=12)
ax1.set_title('DynamicCompactDetect vs YOLOv8: Accuracy Comparison', fontsize=14)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_ylim(35, 55)

# Add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=10)

# Create the FPS plot
bars2 = ax2.bar(models, fps_values, color=colors, width=0.6, edgecolor='black', hatch=hatches)
ax2.set_ylabel('FPS (higher is better)', fontsize=12)
ax2.set_title('DynamicCompactDetect vs YOLOv8: Speed Comparison', fontsize=14)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_ylim(0, 1000)

# Add value labels on top of bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 15,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=10)

# Add a legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='YOLOv8'),
    Patch(facecolor='#e74c3c', edgecolor='black', hatch='///', label='DynamicCompactDetect')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
           fancybox=True, shadow=True, ncol=2)

plt.xticks(rotation=45)
plt.xlabel('Model Variant', fontsize=12)

# Add watermark
fig.text(0.5, 0.01, 'DynamicCompactDetect Research Paper', 
         ha='center', va='bottom', fontsize=8, alpha=0.5)

# Save figure
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'performance_comparison.png'), 
            dpi=300, bbox_inches='tight')
print(f"Saved performance comparison chart to {os.path.dirname(os.path.abspath(__file__))}/performance_comparison.png") 