import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Data for plotting
models = ['YOLOv8-n', 'DCD-S', 'YOLOv8-s', 'DCD-M', 'YOLOv8-m', 'DCD-L', 'YOLOv8-l']
size_mb = [6.3, 21.8, 22.6, 49.5, 52.2, 82.1, 87.7]  # Model size in MB
efficiency = [5.92, 2.09, 1.99, 1.03, 0.96, 0.65, 0.60]  # mAP/MB (derived from mAP divided by model size)

# Setting up the figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.tight_layout(pad=5.0)

# Define colors for different model families
colors = ['#3498db', '#e74c3c', '#3498db', '#e74c3c', '#3498db', '#e74c3c', '#3498db']
hatches = ['', '///', '', '///', '', '///', '']

# Create the model size plot
bars1 = ax1.bar(models, size_mb, color=colors, width=0.6, edgecolor='black', hatch=hatches)
ax1.set_ylabel('Model Size (MB)', fontsize=12)
ax1.set_title('DynamicCompactDetect vs YOLOv8: Model Size Comparison', fontsize=14)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
             f'{height:.1f}',
             ha='center', va='bottom', fontsize=10)

# Create the efficiency plot
bars2 = ax2.bar(models, efficiency, color=colors, width=0.6, edgecolor='black', hatch=hatches)
ax2.set_ylabel('Efficiency (mAP/MB)', fontsize=12)
ax2.set_title('DynamicCompactDetect vs YOLOv8: Efficiency Comparison', fontsize=14)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.08,
             f'{height:.2f}',
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

# Annotate key insights
ax1.annotate('DCD models are slightly smaller than equivalent YOLOv8 models', 
            xy=(4, 70), xytext=(4, 75),
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7),
            fontsize=9)

ax2.annotate('DCD models achieve higher efficiency (more mAP per MB)',
            xy=(3, 1.5), xytext=(3, 1.8),
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7),
            fontsize=9)

# Save figure
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_size_comparison.png'), 
            dpi=300, bbox_inches='tight')
print(f"Saved model size comparison chart to {os.path.dirname(os.path.abspath(__file__))}/model_size_comparison.png") 