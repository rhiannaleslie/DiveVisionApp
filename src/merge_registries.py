import json
from src.constants import (
    VIDEO_REGISTRY_JSON,
    CHUNK_REGISTRY_JSON,
    CAPTION_REGISTRY_JSON,
    MERGED_REGISTRY_JSON,
)


def merge_registries() -> dict:
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
    with open(MERGED_REGISTRY_JSON, "w") as f:
        json.dump(merged, f, indent=2)


if __name__ == "__main__":
    merged = merge_registries()
    save_merged_registry(merged)
    print(f"Merged {len(merged)} chunks → {MERGED_REGISTRY_JSON}")
