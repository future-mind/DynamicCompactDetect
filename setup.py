from setuptools import setup, find_packages

setup(
    name="dynamiccompact",
    version="0.1.0",
    description="DynamicCompact-Detect: A lightweight object detection model",
    author="Future Mind",
    author_email="contact@future-mind.io",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "numpy>=1.18.5",
        "opencv-python>=4.1.2",
        "pyyaml>=5.3.1",
        "matplotlib>=3.2.2",
        "tqdm>=4.41.0",
        "Pillow>=7.1.2",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 