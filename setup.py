from setuptools import setup, find_packages

setup(
    name="dynamiccompactdetect",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    description="Dynamic Compact Object Detection Model",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pillow>=9.4.0",
        "opencv-python>=4.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "pycocotools>=2.0.6",
        "albumentations>=1.3.0",
    ],
    python_requires=">=3.8",
) 