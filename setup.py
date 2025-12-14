#!/usr/bin/env python3
"""
Setup script for Video Identity Transformation System.
"""

from setuptools import setup, find_packages

setup(
    name="video-identity-transform",
    version="0.1.0",
    description="AI-powered video identity transformation with expression and lip sync preservation",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
    ],
    extras_require={
        "full": [
            "ultralytics>=8.0.0",
            "mediapipe>=0.10.0",
            "insightface>=0.7.3",
            "diffusers>=0.25.0",
            "transformers>=4.35.0",
            "accelerate>=0.25.0",
            "controlnet-aux>=0.0.7",
            "segment-anything",
        ]
    },
    entry_points={
        "console_scripts": [
            "video-transform=main:main",
        ],
    },
)

