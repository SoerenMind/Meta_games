"""Package setup script."""
import sys

import setuptools

setuptools.setup(
    name="metagames",
    version="0.1",
    description="Open Source Games with Neural Networks",
    author="Sören Mindermann",
    author_email="soeren.mindermann@gmail.com",
    packages=setuptools.find_packages(),
    install_requires=[
        "gym>=0.10.5",
        "matplotlib>=2.1.2",
        "numpy>=1.14.0",
        "torch>=0.4.0a0",
        "torchvision>=0.2",
    ],
    extras_require={},
)
