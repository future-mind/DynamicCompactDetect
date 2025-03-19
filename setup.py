from setuptools import setup, find_packages

setup(
    name="dynamiccompactdetect",
    version="0.1.0",
    description="A novel dynamic compact detection model for object detection",
    author="Author",
    author_email="author@example.com",
    url="https://github.com/yourusername/DynamicCompactDetect",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
        "pycocotools>=2.0.5",
        "tqdm>=4.60.0",
        "tensorboard>=2.10.0",
        "pillow>=9.0.0",
        "albumentations>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 