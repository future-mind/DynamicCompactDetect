#!/usr/bin/env python3
"""
Generate figures for the DynamicCompactDetect research paper.
This script creates visualization for:
1. Architecture diagram
2. Early exit distribution
3. Performance comparison charts
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patches as patches

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_architecture_diagram():
    """Generate the DynamicCompactDetect architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define component boxes
    components = [
        {"name": "Input", "pos": (5, 9), "width": 2, "height": 0.8, "color": "lightgray"},
        {"name": "Backbone with\nDynamic Blocks", "pos": (5, 7.5), "width": 2, "height": 1.2, "color": "lightblue"},
        {"name": "Early Exit Branch 1", "pos": (8, 6.5), "width": 2, "height": 0.8, "color": "lightgreen"},
        {"name": "Selective Routing", "pos": (5, 5.5), "width": 2, "height": 1, "color": "lightsalmon"},
        {"name": "Early Exit Branch 2", "pos": (8, 4.5), "width": 2, "height": 0.8, "color": "lightgreen"},
        {"name": "Feature Pyramid\nNetwork", "pos": (5, 3.5), "width": 2, "height": 1, "color": "plum"},
        {"name": "Detection Heads", "pos": (5, 1.5), "width": 2, "height": 1, "color": "lightcoral"},
        {"name": "First Exit\nPredictions", "pos": (11, 6.5), "width": 2, "height": 0.8, "color": "paleturquoise"},
        {"name": "Second Exit\nPredictions", "pos": (11, 4.5), "width": 2, "height": 0.8, "color": "paleturquoise"},
        {"name": "Final Predictions", "pos": (5, 0.5), "width": 2, "height": 0.8, "color": "paleturquoise"}
    ]
    
    # Draw component boxes
    for comp in components:
        rect = Rectangle(comp["pos"], comp["width"], comp["height"], 
                        facecolor=comp["color"], edgecolor="black", alpha=0.8)
        ax.add_patch(rect)
        ax.text(comp["pos"][0] + comp["width"]/2, comp["pos"][0] + comp["height"]/2, 
                comp["name"], ha="center", va="center", fontsize=9)
    
    # Draw arrows
    arrows = [
        {"start": (6, 9), "end": (6, 8.7)},  # Input to Backbone
        {"start": (6, 7.5), "end": (6, 6.5)},  # Backbone to Selective Routing
        {"start": (6, 5.5), "end": (6, 4.5)},  # Selective Routing to FPN
        {"start": (6, 3.5), "end": (6, 2.5)},  # FPN to Detection Heads
        {"start": (6, 1.5), "end": (6, 1.3)},  # Detection Heads to Final Predictions
        
        # Early exit branches
        {"start": (7, 7), "end": (8, 6.9), "style": "arc3,rad=0.1"},  # Backbone to Early Exit 1
        {"start": (10, 6.9), "end": (11, 6.9)},  # Early Exit 1 to First Exit Predictions
        {"start": (7, 5), "end": (8, 4.9), "style": "arc3,rad=0.1"},  # Selective Routing to Early Exit 2
        {"start": (10, 4.9), "end": (11, 4.9)},  # Early Exit 2 to Second Exit Predictions
    ]
    
    for arrow in arrows:
        style = arrow.get("style", "simple")
        if style == "simple":
            ax.arrow(arrow["start"][0], arrow["start"][1], 
                    arrow["end"][0] - arrow["start"][0], 
                    arrow["end"][1] - arrow["start"][1],
                    head_width=0.1, head_length=0.1, fc='black', ec='black')
        else:
            arrow_patch = FancyArrowPatch(arrow["start"], arrow["end"], 
                                         connectionstyle=style,
                                         arrowstyle='->', linewidth=1.5, color='black')
            ax.add_patch(arrow_patch)
    
    # Add title and clean up
    ax.set_title("DynamicCompactDetect Architecture", fontsize=14)
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('figures/architecture_diagram.png', dpi=300)
    plt.close()
    print("Generated architecture diagram: figures/architecture_diagram.png")

def generate_early_exit_distribution():
    """Generate the early exit distribution chart."""
    exit_points = ['First Exit', 'Second Exit', 'Complete Inference']
    percentages = [35, 28, 37]
    colors = ['lightgreen', 'lightsalmon', 'lightblue']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(exit_points, percentages, color=colors, width=0.6, edgecolor='black')
    
    # Add labels on top of bars
    for bar, percentage in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{percentage}%', ha='center', va='bottom', fontsize=12)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize plot
    ax.set_title('Exit Distribution for DCD-M on COCO Validation Set', fontsize=14)
    ax.set_ylabel('Percentage of Inputs', fontsize=12)
    ax.set_ylim(0, 50)  # Set y-axis limit
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add annotations
    ax.text(0, 42, 'Simple scenes with few objects', fontsize=10, ha='center')
    ax.text(1, 35, 'Moderately complex scenes', fontsize=10, ha='center')
    ax.text(2, 44, 'Complex scenes with\nmultiple objects or occlusions', fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('figures/early_exit_distribution.png', dpi=300)
    plt.close()
    print("Generated early exit distribution: figures/early_exit_distribution.png")

def generate_performance_comparison():
    """Generate performance comparison chart between DCD and YOLOv8."""
    models = ['YOLOv8-s', 'DCD-S', 'YOLOv8-m', 'DCD-M', 'YOLOv8-l', 'DCD-L']
    fps = [428, 512, 232, 305, 165, 196]
    map_values = [44.9, 45.6, 50.2, 50.9, 52.9, 53.2]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Inference Speed (FPS)
    colors = ['lightgray', 'lightblue', 'lightgray', 'lightblue', 'lightgray', 'lightblue']
    ax1.bar(models, fps, color=colors, edgecolor='black')
    ax1.set_title('Inference Speed (FPS)', fontsize=14)
    ax1.set_ylabel('Frames Per Second', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    # mAP Values
    ax2.bar(models, map_values, color=colors, edgecolor='black')
    ax2.set_title('Detection Accuracy (mAP@0.5:0.95)', fontsize=14)
    ax2.set_ylabel('mAP', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_ylim(40, 55)  # Set y-axis limit for better visualization
    
    # Rotate x-axis labels for better readability
    ax2.set_xticklabels(models, rotation=45, ha='right')
    
    # Add a note
    fig.text(0.5, 0.01, 'DynamicCompactDetect achieves higher mAP with faster inference across size variants', 
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for the note
    plt.savefig('figures/performance_comparison.png', dpi=300)
    plt.close()
    print("Generated performance comparison: figures/performance_comparison.png")

def generate_cross_platform_speedups():
    """Generate cross-platform speedup comparison chart."""
    platforms = ['NVIDIA RTX 3090', 'NVIDIA RTX 2080 Ti', 'M1 Max', 'M2 Ultra', 'Intel i9-12900K']
    speedups = [1.31, 1.30, 1.51, 1.80, 1.47]
    colors = ['#72b7b2', '#72b7b2', '#f17e7e', '#f17e7e', '#b7b272']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(platforms, speedups, color=colors, edgecolor='black')
    
    # Add labels on top of bars
    for bar, speedup in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=12)
    
    # Customize plot
    ax.set_title('DCD-M Speedup Compared to YOLOv8-m Across Platforms', fontsize=14)
    ax.set_ylabel('Speedup (×)', fontsize=12)
    ax.set_ylim(1, 2.0)  # Set y-axis limit
    ax.axhline(y=1, color='black', linestyle='-', alpha=0.3)  # Add baseline at y=1
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Color-coded legend for platform types
    handles = [
        Rectangle((0,0), 1, 1, color='#72b7b2', ec='black'),
        Rectangle((0,0), 1, 1, color='#f17e7e', ec='black'),
        Rectangle((0,0), 1, 1, color='#b7b272', ec='black')
    ]
    labels = ['NVIDIA GPUs', 'Apple Silicon', 'Intel CPU']
    ax.legend(handles, labels, loc='upper right')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('figures/cross_platform_speedups.png', dpi=300)
    plt.close()
    print("Generated cross-platform speedups: figures/cross_platform_speedups.png")

def main():
    # Ensure the figures directory exists
    ensure_dir('figures')
    
    # Generate all figures
    generate_architecture_diagram()
    generate_early_exit_distribution()
    generate_performance_comparison()
    generate_cross_platform_speedups()
    
    print("All figures generated successfully.")

if __name__ == "__main__":
    main() 