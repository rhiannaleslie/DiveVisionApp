import re
from pathlib import Path


def get_video_files(directory: Path) -> list[Path]:
    """Return all video files in a directory.

    Args:
        directory: Path to search for video files.

    Returns:
        Sorted list of video file paths.
    """
    extensions = {".mp4", ".mov", ".avi", ".mkv"}
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in extensions)


def sanitise_filename(name: str) -> str:
    """Strip characters unsafe for filenames.

    Args:
        name: Raw string to sanitise.

    Returns:
        Filename-safe string.
    """
    return re.sub(r"[^\w\-_.]", "_", name)
