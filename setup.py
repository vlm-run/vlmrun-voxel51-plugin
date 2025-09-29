"""
Setup script for VLM Run FiftyOne plugin.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vlmrun-fiftyone-plugin",
    version="1.0.0",
    author="VLM Run Team",
    author_email="support@vlm.run",
    description="FiftyOne plugin for VLM Run's vision-language model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vlm-run/vlmrun-voxel51-plugin",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "fiftyone.plugins": [
            "vlmrun = vlmrun",
        ],
    },
    include_package_data=True,
    package_data={
        "vlmrun": ["fiftyone.yml"],
    },
)
