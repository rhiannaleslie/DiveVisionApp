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


def extract_metadata(video_path: Path) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

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


def extract_file_metadata(video_path: Path) -> dict:
    """Extract created_on and modified_on dates via hachoir. Returns None for missing fields."""
    result = {"created_on": None, "modified_on": None}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = createParser(str(video_path))
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


def _parse_gpsu(raw: bytes) -> str | None:
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


def _parse_gps5(data: bytes) -> list[dict]:
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


def _parse_scalar(data: bytes, key: bytes, type_char: str, scale: float) -> list[float]:
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


def extract_gpmf_metadata(video_path: Path) -> dict:
    """Parse GoPro GPMF stream for GPS ranges, GPS timestamps, and ISO range.

    Returns None for all fields if the file has no GPMF stream (e.g. non-GoPro footage).
    """
    result = {
        "gps_lat_min": None, "gps_lat_max": None,
        "gps_lon_min": None, "gps_lon_max": None,
        "gps_start_ts": None, "gps_end_ts": None,
        "iso_min": None, "iso_max": None,
    }
    try:
        data = video_path.read_bytes()

        gpsu_pos = data.find(b"GPSU")
        if gpsu_pos < 0:
            return result

        result["gps_start_ts"] = _parse_gpsu(data[gpsu_pos + 8: gpsu_pos + 24])
        last_gpsu_pos = data.rfind(b"GPSU")
        result["gps_end_ts"] = _parse_gpsu(data[last_gpsu_pos + 8: last_gpsu_pos + 24])

        gps_samples = _parse_gps5(data)
        if gps_samples:
            lats = [s["lat"] for s in gps_samples]
            lons = [s["lon"] for s in gps_samples]
            result["gps_lat_min"] = round(min(lats), 6)
            result["gps_lat_max"] = round(max(lats), 6)
            result["gps_lon_min"] = round(min(lons), 6)
            result["gps_lon_max"] = round(max(lons), 6)

        isos = _parse_scalar(data, b"ISOE", "S", 1.0)
        if isos:
            result["iso_min"] = int(min(isos))
            result["iso_max"] = int(max(isos))
    except Exception:
        pass
    return result


def load_registry() -> dict:
    if VIDEO_REGISTRY_JSON.exists():
        return json.loads(VIDEO_REGISTRY_JSON.read_text())
    return {}


def save_registry(registry: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_REGISTRY_JSON.write_text(json.dumps(registry, indent=2))


def register_video(video_path: Path) -> str:
    video_path = video_path.resolve()
    registry = load_registry()

    for video_id, record in registry.items():
        if record["filepath"] == str(video_path):
            print(f"  Already registered: {video_path.name} (video_id={video_id})")
            return video_id

    print(f"  Extracting metadata from {video_path.name}...")
    metadata = extract_metadata(video_path)
    file_meta = extract_file_metadata(video_path)
    gpmf_meta = extract_gpmf_metadata(video_path)

    video_id = str(uuid.uuid4())
    registry[video_id] = {
        "video_id": video_id,
        "filename": video_path.name,
        "filepath": str(video_path),
        **metadata,
        **file_meta,
        **gpmf_meta,
    }

    save_registry(registry)

    print(f"    Registered {video_path.name}")
    print(f"    video_id   : {video_id}")
    print(f"    resolution : {metadata['width']}x{metadata['height']}")
    print(f"    fps        : {metadata['fps']}")
    print(f"    duration   : {metadata['duration_s']}s ({metadata['frame_count']} frames)")
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
