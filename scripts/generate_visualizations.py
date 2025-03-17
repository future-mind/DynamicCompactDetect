#!/usr/bin/env python
# Generate visualizations for DynamicCompactDetect performance report

import matplotlib.pyplot as plt
import numpy as np
import os

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Model performance data
models = ['YOLOv8n', 'DynamicCompactDetect']

# Inference time comparison
inference_times = [19.81, 21.07]
plt.figure(figsize=(10, 6))
bars = plt.bar(models, inference_times, color=['#3498db', '#e74c3c'])
plt.title('Average Inference Time (lower is better)')
plt.ylabel('Time (ms)')
plt.ylim(0, max(inference_times) * 1.2)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.2f} ms', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('visualizations/inference_time_comparison.png')
plt.close()

# Detection count comparison
detection_counts = [4.5, 4.0]
plt.figure(figsize=(10, 6))
bars = plt.bar(models, detection_counts, color=['#3498db', '#e74c3c'])
plt.title('Average Detections per Image')
plt.ylabel('Number of Detections')
plt.ylim(0, max(detection_counts) * 1.2)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('visualizations/detection_count_comparison.png')
plt.close()

# COCO metrics comparison
metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
yolov8n_values = [0.5212, 0.3764, 0.5514, 0.4024]
dcd_values = [0.5783, 0.3929, 0.6689, 0.3895]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, yolov8n_values, width, label='YOLOv8n', color='#3498db')
rects2 = ax.bar(x + width/2, dcd_values, width, label='DynamicCompactDetect', color='#e74c3c')

ax.set_title('COCO Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Add percentage labels
def autolabel(rects1, rects2):
    for i, (rect1, rect2) in enumerate(zip(rects1, rects2)):
        height1 = rect1.get_height()
        height2 = rect2.get_height()
        
        # Calculate percentage difference
        if height1 > 0:
            pct_diff = ((height2 - height1) / height1) * 100
            label = f"{'+' if pct_diff > 0 else ''}{pct_diff:.1f}%"
        else:
            label = "N/A"
            
        # Only add percentage label to the higher bar
        if height2 >= height1:
            ax.annotate(label,
                        xy=(rect2.get_x() + rect2.get_width() / 2, height2),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color='green' if pct_diff > 0 else 'red')
        else:
            ax.annotate(label,
                        xy=(rect1.get_x() + rect1.get_width() / 2, height1),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color='green' if pct_diff > 0 else 'red')

        # Add actual values to each bar
        ax.annotate(f'{height1:.4f}',
                    xy=(rect1.get_x() + rect1.get_width() / 2, height1 / 2),
                    ha='center', va='center',
                    color='white', fontweight='bold')
        ax.annotate(f'{height2:.4f}',
                    xy=(rect2.get_x() + rect2.get_width() / 2, height2 / 2),
                    ha='center', va='center',
                    color='white', fontweight='bold')

autolabel(rects1, rects2)

fig.tight_layout()
plt.savefig('visualizations/coco_metrics_comparison.png')
plt.close()

# Combined radar chart
plt.figure(figsize=(10, 10))
categories = ['Inference Speed', 'mAP50', 'Precision', 'Recall', 'Model Size']

# Normalize values for radar chart (higher is better for all metrics)
# For inference time, we invert the value so higher is better
yolov8n_radar = [
    1 / (19.81 / min(19.81, 21.07)),  # Inference speed (inverted time)
    0.5212 / max(0.5212, 0.5783),     # mAP50
    0.5514 / max(0.5514, 0.6689),     # Precision
    0.4024 / max(0.4024, 0.3895),     # Recall
    1.0                              # Model size (same for both)
]

dcd_radar = [
    1 / (21.07 / min(19.81, 21.07)),  # Inference speed (inverted time)
    0.5783 / max(0.5212, 0.5783),     # mAP50
    0.6689 / max(0.5514, 0.6689),     # Precision
    0.3895 / max(0.4024, 0.3895),     # Recall
    1.0                              # Model size (same for both)
]

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Close the loop

yolov8n_radar += yolov8n_radar[:1]  # Close the loop
dcd_radar += dcd_radar[:1]  # Close the loop

categories += categories[:1]  # Close the loop

ax = plt.subplot(111, polar=True)
ax.plot(angles, yolov8n_radar, 'o-', linewidth=2, label='YOLOv8n', color='#3498db')
ax.fill(angles, yolov8n_radar, alpha=0.25, color='#3498db')
ax.plot(angles, dcd_radar, 'o-', linewidth=2, label='DynamicCompactDetect', color='#e74c3c')
ax.fill(angles, dcd_radar, alpha=0.25, color='#e74c3c')

ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
ax.set_ylim(0, 1.1)
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Model Performance Comparison (normalized)')

plt.tight_layout()
plt.savefig('visualizations/radar_comparison.png')
plt.close()

print("Visualizations generated and saved to 'visualizations/' directory") 