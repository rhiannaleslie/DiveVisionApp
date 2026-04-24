import math
from pathlib import Path

import cv2

from src.constants import FRAMES_DIR
from src.chunk_video import CHUNK_REGISTRY_JSON
from src.ingestion import load_registry

SAMPLE_INTERVAL_S = 3.33


def sample_timestamps(chunk: dict, interval_s: float = SAMPLE_INTERVAL_S) -> list[float]:
    """Return the timestamps (in seconds) to sample within a chunk.

    Sampling starts at chunk start and steps by interval_s, always including
    the first frame. The last sample is dropped if it would land past the end.

    Args:
        chunk: A chunk record from the chunk registry.
        interval_s: Desired gap between sampled frames in seconds.

    Returns:
        List of absolute timestamps in seconds.
    """
    start = chunk["start_s"]
    end = chunk["end_s"]
    num_samples = math.floor((end - start) / interval_s) + 1
    timestamps = [round(start + i * interval_s, 3) for i in range(num_samples)]
    # Drop any that overshoot the chunk end
    return [t for t in timestamps if t < end]


def extract_frame(video_path: Path, timestamp_s: float) -> cv2.typing.MatLike:
    """Read a single frame at the given timestamp from a video file.

    Args:
        video_path: Path to the video file.
        timestamp_s: Time in seconds to seek to.

    Returns:
        BGR image array (numpy ndarray).

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the video cannot be opened or the frame cannot be read.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read frame at {timestamp_s}s from {video_path.name}")
        return frame
    finally:
        cap.release()


def chunk_output_dir(chunk: dict, video_filename: str, base_dir: Path = FRAMES_DIR) -> Path:
    """Return the output directory for a chunk's frames.

    Structure: <base_dir>/<video_stem>/chunk_<index:03d>/

    Args:
        chunk: A chunk record from the chunk registry.
        video_filename: The filename of the source video (used to derive folder name).
        base_dir: Root directory under which frames are saved.

    Returns:
        Path to the chunk's output directory (not yet created).
    """
    video_stem = Path(video_filename).stem
    out_dir = (base_dir / video_stem / f"chunk_{chunk['chunk_index']:03d}").resolve()
    if not out_dir.is_relative_to(base_dir.resolve()):
        raise ValueError(f"Resolved path escapes base directory: {out_dir}")
    return out_dir


def extract_chunk_frames(
    chunk: dict,
    video_record: dict,
    interval_s: float = SAMPLE_INTERVAL_S,
    base_dir: Path = FRAMES_DIR,
) -> list[Path]:
    """Extract sampled frames for a single chunk and save them as JPEGs.

    Args:
        chunk: A chunk record from the chunk registry.
        video_record: The corresponding video record from the video registry.
        interval_s: Sampling interval in seconds.
        base_dir: Root directory under which frames are saved.

    Returns:
        List of Paths to the saved JPEG files.
    """
    video_path = Path(video_record["filepath"])
    out_dir = chunk_output_dir(chunk, video_record["filename"], base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps = sample_timestamps(chunk, interval_s)
    saved_paths = []

    for t in timestamps:
        frame = extract_frame(video_path, t)
        filename = f"frame_{t:.3f}s.jpg"
        out_path = out_dir / filename
        cv2.imwrite(str(out_path), frame)
        saved_paths.append(out_path)

    return saved_paths


def extract_all_frames(
    interval_s: float = SAMPLE_INTERVAL_S,
    base_dir: Path = FRAMES_DIR,
    video_ids: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Extract sampled frames for all chunks across all registered videos.

    Args:
        interval_s: Sampling interval in seconds.
        base_dir: Root directory under which frames are saved.
        video_ids: Optional list of video_ids to restrict processing to.

    Returns:
        Dict mapping chunk_id -> list of saved frame Paths.
    """
    import json

    base_dir.mkdir(parents=True, exist_ok=True)

    video_registry = load_registry()
    chunk_registry: dict = json.loads(CHUNK_REGISTRY_JSON.read_text())

    if video_ids is not None:
        chunk_registry = {
            cid: c for cid, c in chunk_registry.items()
            if c["video_id"] in video_ids
        }

    results: dict[str, list[Path]] = {}

    for chunk_id, chunk in chunk_registry.items():
        video_record = video_registry[chunk["video_id"]]
        saved = extract_chunk_frames(chunk, video_record, interval_s, base_dir)
        results[chunk_id] = saved

    return results
