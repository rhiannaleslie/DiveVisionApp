"""
DiveVision ingestion pipeline CLI.

Runs the full pipeline (or individual steps) to ingest raw dive videos,
extract and caption frames, and embed everything into a ChromaDB vector store
ready for semantic retrieval.

Usage examples:

  # Full pipeline — scans data/raw/ automatically
  python -m src.ingestion_pipeline.cli run

  # Full pipeline on specific files
  python -m src.ingestion_pipeline.cli run --videos data/raw/GX014538.MP4

  # Individual steps
  python -m src.ingestion_pipeline.cli ingest --videos data/raw/GX014538.MP4
  python -m src.ingestion_pipeline.cli chunk --duration 20
  python -m src.ingestion_pipeline.cli frames --interval 3.33
  python -m src.ingestion_pipeline.cli caption
  python -m src.ingestion_pipeline.cli merge
  python -m src.ingestion_pipeline.cli embed
"""

import argparse
import sys
from pathlib import Path

from src.constants import RAW_VIDEO_DIR, FRAMES_DIR
from src.ingestion_pipeline.ingestion import VideoFile
from src.ingestion_pipeline.chunk_video import ChunkRegistry, CHUNK_DURATION_S
from src.ingestion_pipeline.extract_frames import FrameExtractor, SAMPLE_INTERVAL_S
from src.ingestion_pipeline.caption_frames import caption_all_chunks
from src.ingestion_pipeline.merge_registries import merge_registries, save_merged_registry
from src.ingestion_pipeline.vector_store import ChunkVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_videos(args) -> list[Path]:
    """Return the list of video paths to process.

    Uses --videos if provided, otherwise scans RAW_VIDEO_DIR for all MP4 files.
    Exits with an error message if no videos are found.
    """
    if args.videos:
        paths = [Path(v) for v in args.videos]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"Error: video file(s) not found: {[str(p) for p in missing]}", file=sys.stderr)
            sys.exit(1)
        return paths

    paths = sorted(RAW_VIDEO_DIR.glob("*.MP4")) + sorted(RAW_VIDEO_DIR.glob("*.mp4"))
    if not paths:
        print(f"Error: no MP4 files found in {RAW_VIDEO_DIR}", file=sys.stderr)
        sys.exit(1)
    return paths


# ---------------------------------------------------------------------------
# Step handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args) -> None:
    """Register each video file in the video registry."""
    videos = _resolve_videos(args)
    print(f"\n=== Ingest ({len(videos)} video(s)) ===")
    for path in videos:
        VideoFile(path).register()


def cmd_chunk(args) -> None:
    """Divide all registered videos into fixed-length chunks."""
    print(f"\n=== Chunk (duration={args.duration}s) ===")
    registry = ChunkRegistry()
    registry.build(chunk_duration_s=args.duration)
    out = registry.save()
    chunks = registry.all()
    n_videos = len({c["video_id"] for c in chunks.values()})
    print(f"  {len(chunks)} chunks across {n_videos} video(s) → {out}")


def cmd_frames(args) -> None:
    """Extract sampled frames from each chunk."""
    output_dir = Path(args.output_dir) if args.output_dir else FRAMES_DIR
    print(f"\n=== Extract frames (interval={args.interval}s, output={output_dir}) ===")
    extractor = FrameExtractor(output_dir=output_dir, interval_s=args.interval)
    results = extractor.extract_all()
    total = sum(len(v) for v in results.values())
    print(f"  {total} frames across {len(results)} chunks")


def cmd_caption(args) -> None:
    """Caption every extracted frame using BLIP."""
    print("\n=== Caption frames ===")
    caption_all_chunks()


def cmd_merge(args) -> None:
    """Merge video, chunk, and caption registries into merged_registry.json."""
    print("\n=== Merge registries ===")
    merged = merge_registries()
    save_merged_registry(merged)
    print(f"  {len(merged)} chunks merged → merged_registry.json")


def cmd_embed(args) -> None:
    """Embed all chunks into the ChromaDB vector store."""
    print("\n=== Embed into vector store ===")
    store = ChunkVectorStore()
    store.update()
    print(f"  Total indexed: {store.count()} chunks")


def cmd_run(args) -> None:
    """Run all pipeline steps end to end."""
    print("=== DiveVision ingestion pipeline ===")
    cmd_ingest(args)
    cmd_chunk(args)
    cmd_frames(args)
    cmd_caption(args)
    cmd_merge(args)
    cmd_embed(args)
    print("\nPipeline complete.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divevision-ingest",
        description="DiveVision ingestion pipeline — ingest, chunk, caption, and embed dive videos.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    videos_kwargs = dict(
        nargs="+",
        metavar="PATH",
        default=None,
        help="Video file path(s). Defaults to all MP4s in data/raw/.",
    )

    # run
    run_p = subparsers.add_parser("run", help="Run the full pipeline end to end")
    run_p.add_argument("--videos", **videos_kwargs)
    run_p.add_argument("--duration", type=float, default=CHUNK_DURATION_S, metavar="S",
                       help=f"Chunk duration in seconds (default: {CHUNK_DURATION_S})")
    run_p.add_argument("--interval", type=float, default=SAMPLE_INTERVAL_S, metavar="S",
                       help=f"Frame sampling interval in seconds (default: {SAMPLE_INTERVAL_S})")
    run_p.add_argument("--output-dir", metavar="DIR", default=None,
                       help="Root directory for extracted frames (default: data/processed/frames/)")
    run_p.set_defaults(func=cmd_run)

    # ingest
    ingest_p = subparsers.add_parser("ingest", help="Register video files in the video registry")
    ingest_p.add_argument("--videos", **videos_kwargs)
    ingest_p.set_defaults(func=cmd_ingest)

    # chunk
    chunk_p = subparsers.add_parser("chunk", help="Divide registered videos into chunks")
    chunk_p.add_argument("--duration", type=float, default=CHUNK_DURATION_S, metavar="S",
                         help=f"Chunk duration in seconds (default: {CHUNK_DURATION_S})")
    chunk_p.set_defaults(func=cmd_chunk)

    # frames
    frames_p = subparsers.add_parser("frames", help="Extract sampled frames from each chunk")
    frames_p.add_argument("--interval", type=float, default=SAMPLE_INTERVAL_S, metavar="S",
                          help=f"Sampling interval in seconds (default: {SAMPLE_INTERVAL_S})")
    frames_p.add_argument("--output-dir", metavar="DIR", default=None,
                          help="Root directory for extracted frames (default: data/processed/frames/)")
    frames_p.set_defaults(func=cmd_frames)

    # caption
    caption_p = subparsers.add_parser("caption", help="Caption extracted frames with BLIP")
    caption_p.set_defaults(func=cmd_caption)

    # merge
    merge_p = subparsers.add_parser("merge", help="Merge video, chunk, and caption registries")
    merge_p.set_defaults(func=cmd_merge)

    # embed
    embed_p = subparsers.add_parser("embed", help="Embed chunks into the ChromaDB vector store")
    embed_p.set_defaults(func=cmd_embed)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
