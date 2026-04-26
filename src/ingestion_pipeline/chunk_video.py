import json
import math
import uuid
from pathlib import Path

from src.constants import PROCESSED_DIR, CHUNK_REGISTRY_JSON
from src.ingestion_pipeline.ingestion import VideoRegistry

CHUNK_DURATION_S = 20.0


class ChunkRegistry:
    """Manages the central chunk_registry.json file that tracks all video chunks."""

    def __init__(self, path: Path = CHUNK_REGISTRY_JSON):
        """Load the registry from disk, or start with an empty registry if it doesn't exist.

        Args:
            path: Path to the JSON registry file. Defaults to CHUNK_REGISTRY_JSON.
        """
        self.path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        """Read the chunk registry JSON from disk.

        Returns:
            Dict of chunk_id -> record, or an empty dict if the file doesn't exist.
        """
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def build(
        self,
        chunk_duration_s: float = CHUNK_DURATION_S,
        video_ids: list[str] | None = None,
    ) -> None:
        """Generate chunk definitions for all (or selected) videos in the video registry.

        Reads the video registry, optionally filters to a subset of video IDs,
        and populates this registry with the computed chunks.

        Args:
            chunk_duration_s: Target chunk length in seconds.
            video_ids: Optional list of video_ids to process; defaults to all registered videos.

        Raises:
            ValueError: If any of the provided video_ids are not found in the video registry.
        """
        video_registry = VideoRegistry()
        all_videos = video_registry.all()

        if video_ids is not None:
            missing = set(video_ids) - set(all_videos)
            if missing:
                raise ValueError(f"video_ids not found in registry: {missing}")
            all_videos = {vid: all_videos[vid] for vid in video_ids}

        self._data = {}
        for video_record in all_videos.values():
            for chunk in self._compute_chunks(video_record, chunk_duration_s):
                self._data[chunk["chunk_id"]] = chunk

    @staticmethod
    def _compute_chunks(video_record: dict, chunk_duration_s: float) -> list[dict]:
        """Divide a single video into fixed-length chunk definitions.

        The final chunk may be shorter than chunk_duration_s if the video length
        is not an exact multiple of the chunk size. Frame boundaries are clamped
        to the actual frame count to avoid out-of-range indices.

        Args:
            video_record: A video registry entry with fps, duration_s, frame_count.
            chunk_duration_s: Target duration of each chunk in seconds.

        Returns:
            List of chunk dicts with keys:
                chunk_id, video_id, chunk_index,
                start_s, end_s, duration_s, start_frame, end_frame.
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

    def save(self) -> Path:
        """Write the current registry state to disk.

        Returns:
            Path where the file was written.
        """
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))
        return self.path

    def all(self) -> dict:
        """Return a copy of all chunk records in the registry.

        Returns:
            Dict of chunk_id -> record for every chunk.
        """
        return dict(self._data)

