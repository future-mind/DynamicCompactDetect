from setuptools import setup, find_packages

# Get the version from the package without importing it
with open('src/dynamiccompactdetect/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"\'')
            break

# Read the long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="dynamiccompactdetect",
    version=version,
    author="Abhilash Chadhar, Divya Athya",
    author_email="abhilashchadhar@example.com",
    description="A lightweight object detection model for edge devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/future-mind/dynamiccompactdetect",
    project_urls={
        "Bug Tracker": "https://github.com/future-mind/dynamiccompactdetect/issues",
        "Documentation": "https://github.com/future-mind/dynamiccompactdetect/docs",
        "Source Code": "https://github.com/future-mind/dynamiccompactdetect",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "pillow>=8.0.0",
        "pyyaml>=5.4.0",
        "ultralytics>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.5b2",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mkdocs>=1.1.0",
            "mkdocs-material>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dcd=dynamiccompactdetect.cli:main",
        ],
    },
) 