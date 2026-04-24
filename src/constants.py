import os
from pathlib import Path
from src.utils import get_secret

ROOT_DIR = Path(__file__).parent.parent

# Data paths
RAW_VIDEO_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Registry
VIDEO_REGISTRY_JSON = PROCESSED_DIR / "video_registry.json"

# Frames
FRAMES_DIR = PROCESSED_DIR / "frames"

# Output paths
OUTPUTS_DIR = ROOT_DIR / "outputs"

# HF Token
HF_TOKEN = get_secret("APIs/hugging_face")