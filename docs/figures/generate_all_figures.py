#!/usr/bin/env python3
import os
import subprocess
import sys

def run_script(script_path, description):
    """Run a Python script with proper error handling"""
    print(f"\n{'='*80}\nGenerating {description}...\n{'='*80}")
    try:
        result = subprocess.run([sys.executable, script_path], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True,
                               check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(e.stderr)
        return False

def main():
    """Run all figure generation scripts"""
    print(f"Generating all figures for the DynamicCompactDetect research paper...")
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create all the necessary directories
    os.makedirs(os.path.join(base_dir, "comparison_charts"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "detection_results"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "model_diagrams"), exist_ok=True)
    
    # List of scripts to run with descriptions
    scripts = [
        (os.path.join(base_dir, "comparison_charts", "performance_comparison.py"), 
         "performance comparison charts"),
        
        (os.path.join(base_dir, "comparison_charts", "model_size_comparison.py"), 
         "model size and efficiency comparison charts"),
        
        (os.path.join(base_dir, "detection_results", "generate_detection_comparisons.py"), 
         "detection comparison visualizations"),
        
        (os.path.join(base_dir, "model_diagrams", "architecture_diagram.py"), 
         "architecture diagram"),
    ]
    
    # Store any failures
    failures = []
    
    # Run each script
    for script_path, description in scripts:
        if not run_script(script_path, description):
            failures.append(script_path)
    
    # Generate a summary of all figures
    print("\n" + "="*80)
    if failures:
        print(f"WARNING: {len(failures)} script(s) failed to run:")
        for script in failures:
            print(f"  - {script}")
    else:
        print("All figure generation scripts completed successfully!")
    
    # Create a summary markdown file listing all generated figures
    generated_figures = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                figure_path = os.path.join(root, file)
                rel_path = os.path.relpath(figure_path, base_dir)
                generated_figures.append((rel_path, os.path.getsize(figure_path)))
    
    # Sort by size (largest first) to highlight important figures
    generated_figures.sort(key=lambda x: x[1], reverse=True)
    
    # Write summary file
    summary_path = os.path.join(base_dir, "figure_summary.md")
    with open(summary_path, 'w') as f:
        f.write("# DynamicCompactDetect Research Paper Figures\n\n")
        f.write("The following figures have been generated for the research paper:\n\n")
        
        # Group by directory
        categories = {}
        for path, size in generated_figures:
            category = os.path.dirname(path) or "Root"
            if category not in categories:
                categories[category] = []
            categories[category].append((path, size))
        
        for category, figures in categories.items():
            f.write(f"## {category.replace('_', ' ').title()}\n\n")
            for path, size in figures:
                size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.1f} MB"
                f.write(f"- `{path}` ({size_str})\n")
            f.write("\n")
        
        f.write("\nThese figures are referenced in the research paper markdown file.\n")
    
    print(f"\nFigure summary written to {summary_path}")
    print("="*80)

if __name__ == "__main__":
    main() 