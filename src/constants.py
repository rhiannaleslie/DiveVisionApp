from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# Data paths
RAW_VIDEO_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Registry
VIDEO_REGISTRY_JSON = PROCESSED_DIR / "video_registry.json"
CHUNK_REGISTRY_JSON = PROCESSED_DIR / "chunk_registry.json"
CAPTION_REGISTRY_JSON = PROCESSED_DIR / "caption_registry.json"
MERGED_REGISTRY_JSON = PROCESSED_DIR / "merged_registry.json"

# Frames
FRAMES_DIR = PROCESSED_DIR / "frames"

# Vector store
VECTOR_STORE_DIR = PROCESSED_DIR / "vector_store"

# Output paths
OUTPUTS_DIR = ROOT_DIR / "outputs"

# HF Token
#HF_TOKEN = get_secret("APIs/hugging_face")