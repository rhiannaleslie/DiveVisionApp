import json

from src.constants import (
    VIDEO_REGISTRY_JSON,
    CHUNK_REGISTRY_JSON,
    CAPTION_REGISTRY_JSON,
    MERGED_REGISTRY_JSON,
)


def merge_registries() -> dict:
    """Combine video, chunk, and caption registries into a single dict keyed by chunk_id.

    Each entry contains the chunk's time boundaries, all its frame captions, and
    the full metadata of the source video.

    Returns:
        Dict of chunk_id -> merged record.

    Raises:
        FileNotFoundError: If any of the three source registries do not exist.
    """
    with open(VIDEO_REGISTRY_JSON) as f:
        videos = json.load(f)
    with open(CHUNK_REGISTRY_JSON) as f:
        chunks = json.load(f)
    with open(CAPTION_REGISTRY_JSON) as f:
        captions = json.load(f)

    merged = {}
    for chunk_id, chunk in chunks.items():
        video = videos.get(chunk["video_id"], {})
        merged[chunk_id] = {
            **chunk,
            "captions": captions.get(chunk_id, {}).get("captions", {}),
            "video": video,
        }

    return merged


def save_merged_registry(merged: dict) -> None:
    """Write the merged registry to disk as JSON.

    Args:
        merged: Dict of chunk_id -> merged record, as returned by merge_registries().
    """
    with open(MERGED_REGISTRY_JSON, "w") as f:
        json.dump(merged, f, indent=2)
