#!/usr/bin/env python3
"""
Model Download Script

Downloads all required models for the video identity transformation system.
"""

import os
import sys
from pathlib import Path
import urllib.request
import hashlib
from tqdm import tqdm


# Model definitions: (name, url, expected_hash, destination)
MODELS = {
    'sam_vit_h': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'path': 'models/sam_vit_h_4b8939.pth',
        'size': '2.4GB'
    },
    'sam_vit_l': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'path': 'models/sam_vit_l_0b3195.pth',
        'size': '1.2GB'
    },
    'sam_vit_b': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'path': 'models/sam_vit_b_01ec64.pth',
        'size': '375MB'
    },
    'inswapper': {
        'url': 'https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx',
        'path': 'models/inswapper_128.onnx',
        'size': '500MB'
    },
    'yolov8x': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
        'path': 'models/yolov8x.pt',
        'size': '130MB'
    }
}

# Hugging Face models (downloaded via transformers/diffusers)
HF_MODELS = [
    'stabilityai/stable-diffusion-xl-base-1.0',
    'thibaud/controlnet-openpose-sdxl-1.0',
    'diffusers/controlnet-canny-sdxl-1.0',
    'diffusers/controlnet-depth-sdxl-1.0',
    'madebyollin/sdxl-vae-fp16-fix',
    'Intel/dpt-hybrid-midas',
    'lllyasviel/ControlNet'
]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, destination: str):
    """Download a file with progress bar."""
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=Path(destination).name) as t:
        urllib.request.urlretrieve(url, destination, reporthook=t.update_to)


def download_direct_models():
    """Download models from direct URLs."""
    print("\n=== Downloading Direct Models ===\n")
    
    for name, info in MODELS.items():
        dest_path = info['path']
        
        if Path(dest_path).exists():
            print(f"✓ {name} already exists at {dest_path}")
            continue
            
        print(f"\nDownloading {name} ({info['size']})...")
        try:
            download_file(info['url'], dest_path)
            print(f"✓ Downloaded {name}")
        except Exception as e:
            print(f"✗ Failed to download {name}: {e}")


def download_hf_models():
    """Download Hugging Face models."""
    print("\n=== Downloading Hugging Face Models ===\n")
    print("These models will be cached in ~/.cache/huggingface/")
    print("This may take a while on first run...\n")
    
    try:
        from huggingface_hub import snapshot_download
        
        for model_id in HF_MODELS:
            print(f"Downloading {model_id}...")
            try:
                snapshot_download(repo_id=model_id, local_files_only=False)
                print(f"✓ {model_id}")
            except Exception as e:
                print(f"✗ Failed: {e}")
                
    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        print("HF models will be downloaded on first use.")


def download_insightface_models():
    """Download InsightFace models."""
    print("\n=== Downloading InsightFace Models ===\n")
    
    try:
        from insightface.app import FaceAnalysis
        
        print("Downloading buffalo_l model...")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("✓ InsightFace models downloaded")
        
    except ImportError:
        print("InsightFace not installed. Install with: pip install insightface")
    except Exception as e:
        print(f"✗ Failed: {e}")


def main():
    print("=" * 60)
    print("Video Identity Transformation - Model Downloader")
    print("=" * 60)
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Download all models
    download_direct_models()
    download_hf_models()
    download_insightface_models()
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nNote: Some models (SDXL, ControlNets) are large and may take")
    print("additional time to download on first use.")


if __name__ == "__main__":
    main()

