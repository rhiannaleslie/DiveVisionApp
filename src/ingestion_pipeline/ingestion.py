import json
import struct
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import cv2
from hachoir.metadata import extractMetadata
from hachoir.parser import createParser

from src.constants import PROCESSED_DIR, VIDEO_REGISTRY_JSON


class VideoFile:
    """Represents a single video file and provides methods to extract its metadata."""

    def __init__(self, video_path: Path):
        """Load a video file, resolving the path to an absolute location.

        Args:
            video_path: Path to the video file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self.path = Path(video_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

    def extract_video_metadata(self) -> dict:
        """Extract core video properties using OpenCV.

        Returns:
            Dict with keys: fps, width, height, duration_s, frame_count.

        Raises:
            ValueError: If OpenCV cannot open the file.
        """
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = round(frame_count / fps, 2) if fps > 0 else 0.0
        cap.release()

        return {
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "duration_s": duration_s,
            "frame_count": frame_count,
        }

    def extract_file_metadata(self) -> dict:
        """Extract file-level timestamps via hachoir.

        Returns:
            Dict with keys: created_on, modified_on (ISO strings, or None if unavailable).
        """
        result = {"created_on": None, "modified_on": None}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parser = createParser(str(self.path))
            if not parser:
                return result
            with parser:
                meta = extractMetadata(parser)
            if not meta:
                return result
            created = meta.get("creation_date")
            modified = meta.get("last_modification")
            if created:
                result["created_on"] = created.isoformat() if hasattr(created, "isoformat") else str(created)
            if modified:
                result["modified_on"] = modified.isoformat() if hasattr(modified, "isoformat") else str(modified)
        except Exception:
            pass
        return result

    def extract_gpmf_metadata(self) -> dict:
        """Parse the GoPro GPMF binary stream for GPS and exposure data.

        Scans the raw file bytes for GPMF telemetry markers. Returns None for
        all fields if no GPMF stream is found (e.g. non-GoPro footage).

        Returns:
            Dict with keys: gps_lat_min, gps_lat_max, gps_lon_min, gps_lon_max,
            gps_start_ts, gps_end_ts, iso_min, iso_max.
        """
        result = {
            "gps_lat_min": None, "gps_lat_max": None,
            "gps_lon_min": None, "gps_lon_max": None,
            "gps_start_ts": None, "gps_end_ts": None,
            "iso_min": None, "iso_max": None,
        }
        try:
            data = self.path.read_bytes()

            gpsu_pos = data.find(b"GPSU")
            if gpsu_pos < 0:
                return result

            result["gps_start_ts"] = self._parse_gpsu(data[gpsu_pos + 8: gpsu_pos + 24])
            last_gpsu_pos = data.rfind(b"GPSU")
            result["gps_end_ts"] = self._parse_gpsu(data[last_gpsu_pos + 8: last_gpsu_pos + 24])

            gps_samples = self._parse_gps5(data)
            if gps_samples:
                lats = [s["lat"] for s in gps_samples]
                lons = [s["lon"] for s in gps_samples]
                result["gps_lat_min"] = round(min(lats), 6)
                result["gps_lat_max"] = round(max(lats), 6)
                result["gps_lon_min"] = round(min(lons), 6)
                result["gps_lon_max"] = round(max(lons), 6)

            isos = self._parse_scalar(data, b"ISOE", "S", 1.0)
            if isos:
                result["iso_min"] = int(min(isos))
                result["iso_max"] = int(max(isos))
        except Exception:
            pass
        return result

    def register(self) -> str:
        """Extract all metadata and write a record to the video registry.

        Skips registration if the file is already present in the registry.

        Returns:
            The UUID string assigned to this video.
        """
        registry = VideoRegistry()
        existing_id = registry.find_by_path(self.path)
        if existing_id:
            print(f"  Already registered: {self.path.name} (video_id={existing_id})")
            return existing_id

        print(f"  Extracting metadata from {self.path.name}...")
        video_meta = self.extract_video_metadata()
        file_meta = self.extract_file_metadata()
        gpmf_meta = self.extract_gpmf_metadata()

        video_id = str(uuid.uuid4())
        record = {
            "video_id": video_id,
            "filename": self.path.name,
            "filepath": str(self.path),
            **video_meta,
            **file_meta,
            **gpmf_meta,
        }
        registry.add(video_id, record)

        print(f"    Registered {self.path.name}")
        print(f"    video_id   : {video_id}")
        print(f"    resolution : {video_meta['width']}x{video_meta['height']}")
        print(f"    fps        : {video_meta['fps']}")
        print(f"    duration   : {video_meta['duration_s']}s ({video_meta['frame_count']} frames)")
        if file_meta["created_on"]:
            print(f"    created_on : {file_meta['created_on']}")
        if gpmf_meta["gps_start_ts"]:
            print(f"    gps_start  : {gpmf_meta['gps_start_ts']}")
            print(f"    gps_end    : {gpmf_meta['gps_end_ts']}")
            print(f"    lat range  : {gpmf_meta['gps_lat_min']} – {gpmf_meta['gps_lat_max']}")
            print(f"    lon range  : {gpmf_meta['gps_lon_min']} – {gpmf_meta['gps_lon_max']}")
        if gpmf_meta["iso_min"] is not None:
            print(f"    ISO range  : {gpmf_meta['iso_min']} – {gpmf_meta['iso_max']}")

        return video_id

    @staticmethod
    def _parse_gpsu(raw: bytes) -> str | None:
        """Decode a GPSU timestamp field into an ISO 8601 string.

        GPSU bytes encode UTC time as ASCII in the format YYMMDDHHMMSSsss,
        with the year offset from 2000.

        Args:
            raw: 16-byte GPSU payload from the GPMF stream.

        Returns:
            ISO 8601 timestamp string, or None if parsing fails.
        """
        try:
            s = raw.decode("ascii").strip()
            dt = datetime(
                2000 + int(s[0:2]), int(s[2:4]), int(s[4:6]),
                int(s[6:8]), int(s[8:10]), int(s[10:12]),
                int(float(s[12:]) * 1_000_000),
                tzinfo=timezone.utc,
            )
            return dt.isoformat()
        except Exception:
            return None

    @staticmethod
    def _parse_gps5(data: bytes) -> list[dict]:
        """Extract all GPS5 samples from the raw GPMF byte stream.

        GPS5 records contain 5 signed 32-bit big-endian integers per sample
        (lat, lon, altitude, 2D speed, 3D speed), scaled by 10^7 for lat/lon.

        Args:
            data: Full raw file bytes.

        Returns:
            List of dicts with 'lat' and 'lon' keys (decimal degrees).
        """
        samples = []
        offset = 0
        while True:
            pos = data.find(b"GPS5", offset)
            if pos < 0:
                break
            type_char = chr(data[pos + 4])
            size = data[pos + 5]
            repeat = struct.unpack(">H", data[pos + 6: pos + 8])[0]
            if type_char == "l" and size == 20 and 0 < repeat < 5_000:
                raw = data[pos + 8: pos + 8 + size * repeat]
                for i in range(repeat):
                    v = struct.unpack(">5i", raw[i * 20: (i + 1) * 20])
                    samples.append({"lat": v[0] / 10_000_000, "lon": v[1] / 10_000_000})
            offset = pos + 1
        return samples

    @staticmethod
    def _parse_scalar(data: bytes, key: bytes, type_char: str, scale: float) -> list[float]:
        """Extract all samples for a named scalar GPMF key.

        Scans the byte stream for every occurrence of `key`, validates the
        type character, and unpacks each repeat value, dividing by `scale`.

        Args:
            data: Full raw file bytes.
            key: 4-byte GPMF key to search for (e.g. b"ISOE").
            type_char: Expected GPMF type character (e.g. "S" for uint16).
            scale: Divisor applied to each raw integer value.

        Returns:
            List of scaled float values across all matching GPMF records.
        """
        fmt_map = {"s": ">h", "S": ">H", "l": ">i", "L": ">I", "f": ">f", "B": "B"}
        if type_char not in fmt_map:
            return []
        fmt = fmt_map[type_char]
        item_size = struct.calcsize(fmt)
        values = []
        offset = 0
        while True:
            pos = data.find(key, offset)
            if pos < 0:
                break
            if chr(data[pos + 4]) == type_char:
                size = data[pos + 5]
                repeat = struct.unpack(">H", data[pos + 6: pos + 8])[0]
                if size == item_size and 0 < repeat < 10_000:
                    raw = data[pos + 8: pos + 8 + size * repeat]
                    for i in range(repeat):
                        (v,) = struct.unpack(fmt, raw[i * item_size: (i + 1) * item_size])
                        values.append(v / scale)
            offset = pos + 1
        return values


class VideoRegistry:
    """Manages the central video_registry.json file that tracks all registered videos."""

    def __init__(self, path: Path = VIDEO_REGISTRY_JSON):
        """Load the registry from disk, or start with an empty registry if it doesn't exist.

        Args:
            path: Path to the JSON registry file. Defaults to VIDEO_REGISTRY_JSON.
        """
        self.path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        """Read the registry JSON from disk.

        Returns:
            Dict of video_id -> record, or an empty dict if the file doesn't exist.
        """
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def _save(self) -> None:
        """Persist the current registry state to disk."""
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    def find_by_path(self, video_path: Path) -> str | None:
        """Look up a video by its absolute file path.

        Args:
            video_path: Absolute path to the video file.

        Returns:
            The video_id string if found, otherwise None.
        """
        for video_id, record in self._data.items():
            if record["filepath"] == str(video_path):
                return video_id
        return None

    def add(self, video_id: str, record: dict) -> None:
        """Add a new record to the registry and save to disk.

        Args:
            video_id: UUID string for the video.
            record: Metadata dict to store under that ID.
        """
        self._data[video_id] = record
        self._save()

    def get(self, video_id: str) -> dict | None:
        """Retrieve a single record by video_id.

        Args:
            video_id: UUID string for the video.

        Returns:
            The metadata dict, or None if the ID is not in the registry.
        """
        return self._data.get(video_id)

    def all(self) -> dict:
        """Return a copy of all records in the registry.

        Returns:
            Dict of video_id -> record for every registered video.
        """
        return dict(self._data)
