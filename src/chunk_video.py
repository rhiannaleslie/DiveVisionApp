import json
import math
import uuid
from pathlib import Path

from src.constants import PROCESSED_DIR
from src.ingestion import load_registry

CHUNK_REGISTRY_JSON = PROCESSED_DIR / "chunk_registry.json"
CHUNK_DURATION_S = 20.0


def chunk_video(video_record: dict, chunk_duration_s: float = CHUNK_DURATION_S) -> list[dict]:
    """Compute chunk definitions for a single video.

    Args:
        video_record: A single entry from the video registry.
        chunk_duration_s: Target duration of each chunk in seconds.

    Returns:
        List of chunk dicts, each with:
            chunk_id, video_id, chunk_index,
            start_s, end_s, start_frame, end_frame, duration_s
    """
    video_id = video_record["video_id"]
    fps = video_record["fps"]
    duration_s = video_record["duration_s"]
    frame_count = video_record["frame_count"]

    num_chunks = math.ceil(duration_s / chunk_duration_s)
    chunks = []

    for i in range(num_chunks):
        start_s = round(i * chunk_duration_s, 3)
        end_s = round(min((i + 1) * chunk_duration_s, duration_s), 3)
        start_frame = min(round(start_s * fps), frame_count - 1)
        end_frame = min(round(end_s * fps), frame_count)

        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "video_id": video_id,
            "chunk_index": i,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": round(end_s - start_s, 3),
            "start_frame": start_frame,
            "end_frame": end_frame,
        })

    return chunks


def build_chunk_registry(
    chunk_duration_s: float = CHUNK_DURATION_S,
    video_ids: list[str] | None = None,
) -> dict:
    """Build chunk definitions for all (or selected) videos in the registry.

    Args:
        chunk_duration_s: Target chunk length in seconds.
        video_ids: Optional list of video_ids to process; defaults to all.

    Returns:
        Dict mapping chunk_id -> chunk record.
    """
    registry = load_registry()

    if video_ids is not None:
        missing = set(video_ids) - set(registry)
        if missing:
            raise ValueError(f"video_ids not found in registry: {missing}")
        registry = {vid: registry[vid] for vid in video_ids}

    chunk_registry: dict = {}
    for video_record in registry.values():
        for chunk in chunk_video(video_record, chunk_duration_s):
            chunk_registry[chunk["chunk_id"]] = chunk

    return chunk_registry


def save_chunk_registry(chunk_registry: dict) -> Path:
    """Write the chunk registry to disk.

    Args:
        chunk_registry: Dict mapping chunk_id -> chunk record.

    Returns:
        Path where the file was written.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_REGISTRY_JSON.write_text(json.dumps(chunk_registry, indent=2))
    return CHUNK_REGISTRY_JSON


if __name__ == "__main__":
    chunk_registry = build_chunk_registry()
    out_path = save_chunk_registry(chunk_registry)

    total_videos = len({c["video_id"] for c in chunk_registry.values()})
    print(f"Chunked {total_videos} video(s) into {len(chunk_registry)} chunks ({CHUNK_DURATION_S}s each)")
    print(f"Saved to {out_path}")
