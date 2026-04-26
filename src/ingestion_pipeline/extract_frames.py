import math
from pathlib import Path

import cv2

from src.constants import FRAMES_DIR
from src.ingestion_pipeline.ingestion import VideoRegistry
from src.ingestion_pipeline.chunk_video import ChunkRegistry

SAMPLE_INTERVAL_S = 3.33


class FrameExtractor:
    """Extracts sampled frames from video chunks and saves them as JPEGs."""

    def __init__(
        self,
        output_dir: Path = FRAMES_DIR,
        interval_s: float = SAMPLE_INTERVAL_S,
    ):
        """Configure the extractor with an output location and sampling rate.

        Args:
            output_dir: Root directory under which all frames will be saved.
                Frames are organised as <output_dir>/<video_stem>/chunk_<index:03d>/.
            interval_s: Gap between sampled frames in seconds.
        """
        self.output_dir = Path(output_dir)
        self.interval_s = interval_s

    def extract_chunk(self, chunk: dict, video_record: dict) -> list[Path]:
        """Extract and save sampled frames for a single chunk.

        Args:
            chunk: A chunk record from the chunk registry.
            video_record: The corresponding video record from the video registry,
                containing at minimum: filepath, filename.

        Returns:
            List of Paths to the saved JPEG files.
        """
        video_path = Path(video_record["filepath"])
        out_dir = self.chunk_output_dir(chunk, video_record["filename"], self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for t in self._sample_timestamps(chunk):
            frame = self._extract_frame(video_path, t)
            out_path = out_dir / f"frame_{t:.3f}s.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_paths.append(out_path)

        return saved_paths

    def extract_all(
        self,
        video_ids: list[str] | None = None,
    ) -> dict[str, list[Path]]:
        """Extract frames for all chunks across all registered videos.

        Reads the video and chunk registries from disk, optionally filtering
        to a subset of videos, then calls extract_chunk for each.

        Args:
            video_ids: Optional list of video_ids to restrict processing to.
                Defaults to all registered videos.

        Returns:
            Dict mapping chunk_id -> list of saved frame Paths.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        video_registry = VideoRegistry()
        chunk_registry = ChunkRegistry()
        all_chunks = chunk_registry.all()

        if video_ids is not None:
            all_chunks = {
                cid: c for cid, c in all_chunks.items()
                if c["video_id"] in video_ids
            }

        results: dict[str, list[Path]] = {}
        for chunk_id, chunk in all_chunks.items():
            video_record = video_registry.get(chunk["video_id"])
            saved = self.extract_chunk(chunk, video_record)
            results[chunk_id] = saved

        return results

    def _sample_timestamps(self, chunk: dict) -> list[float]:
        """Compute the timestamps to sample within a chunk.

        Sampling starts at the chunk start and steps by interval_s. The last
        sample is dropped if it would land at or past the chunk end.

        Args:
            chunk: A chunk record with start_s and end_s keys.

        Returns:
            List of absolute timestamps in seconds.
        """
        start = chunk["start_s"]
        end = chunk["end_s"]
        num_samples = math.floor((end - start) / self.interval_s) + 1
        timestamps = [round(start + i * self.interval_s, 3) for i in range(num_samples)]
        return [t for t in timestamps if t < end]

    def _extract_frame(self, video_path: Path, timestamp_s: float) -> cv2.typing.MatLike:
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

    @staticmethod
    def chunk_output_dir(chunk: dict, video_filename: str, base_dir: Path) -> Path:
        """Resolve the output directory for a chunk's frames.

        Structure: <base_dir>/<video_stem>/chunk_<index:03d>/

        Args:
            chunk: A chunk record with a chunk_index key.
            video_filename: The filename of the source video (stem is used as folder name).
            base_dir: Root directory under which frames are saved.

        Returns:
            Resolved Path to the chunk's output directory (not yet created).

        Raises:
            ValueError: If the resolved path would escape the base directory.
        """
        video_stem = Path(video_filename).stem
        out_dir = (base_dir / video_stem / f"chunk_{chunk['chunk_index']:03d}").resolve()
        if not out_dir.is_relative_to(base_dir.resolve()):
            raise ValueError(f"Resolved path escapes output directory: {out_dir}")
        return out_dir
