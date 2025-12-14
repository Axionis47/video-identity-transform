# Video Identity Transformation System

A comprehensive AI-powered system for transforming person identities in videos while preserving expressions, lip sync, body movements, and environmental interactions.

## Features

- ✅ Multi-person tracking with consistent identity across frames
- ✅ Full-body transformation (not just face swap)
- ✅ Expression and lip sync preservation
- ✅ Gender, race, and appearance transformation
- ✅ Realistic environmental interactions
- ✅ Handles occlusions (people facing toward/away from camera)
- ✅ Temporal consistency across frames

## System Requirements

- **GPU**: NVIDIA GPU with 12GB+ VRAM (24GB recommended for high quality)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space for models
- **OS**: Linux (Ubuntu 20.04+) or Windows with WSL2

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (automated on first run)
python scripts/download_models.py
```

## Quick Start

```bash
# Basic usage
python main.py --input video.mp4 --config configs/transform_config.yaml

# With custom target identities
python main.py --input video.mp4 --targets person1.jpg person2.jpg
```

## Configuration

Edit `configs/transform_config.yaml` to specify:
- Target identities (reference images or text descriptions)
- Quality vs speed tradeoffs
- GPU memory optimization settings

## Architecture

1. **Tracking**: Multi-person detection and tracking (YOLOv8 + ByteTrack)
2. **Analysis**: Pose estimation (MMPose) + facial landmarks (MediaPipe)
3. **Segmentation**: Person segmentation (SAM)
4. **Transformation**: Identity generation (Stable Diffusion + ControlNet)
5. **Compositing**: Scene integration with lighting/shadow adjustment
6. **Temporal**: Frame-to-frame consistency smoothing

## Project Structure

```
├── main.py                 # Main pipeline orchestrator
├── configs/                # Configuration files
├── modules/
│   ├── tracking.py        # Multi-person tracking
│   ├── pose_extraction.py # Pose and expression analysis
│   ├── segmentation.py    # Person segmentation
│   ├── transformation.py  # Identity transformation engine
│   ├── compositing.py     # Scene compositing
│   └── temporal.py        # Temporal consistency
├── models/                 # Downloaded model weights
└── utils/                  # Helper functions
```

## License

Research and educational use only. Ensure you have proper consent for any person appearing in processed videos.

