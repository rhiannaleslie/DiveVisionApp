# DiveVisionApp

App to identify scenes in scuba diving footage.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then:

```bash
uv sync --dev
```

This creates a virtual environment at `.venv/` and installs all dependencies. To activate it:

```bash
source .venv/bin/activate
```

## Ingestion Pipeline

The ingestion pipeline processes raw dive videos into a ChromaDB vector store ready for semantic search. It runs in six steps:

```
ingest → chunk → frames → caption → merge → embed
```

### Run the full pipeline

Place your `.MP4` files in `data/raw/`, then:

```bash
python -m src.ingestion_pipeline.cli run
```

Or target specific files:

```bash
python -m src.ingestion_pipeline.cli run --videos data/raw/GX014538.MP4 data/raw/GX014580.MP4
```

Alternatively, after `pip install -e .` the command is available as:

```bash
divevision-ingest run
```

### Run individual steps

Each step can be run independently, which is useful when adding new videos or re-running a specific stage.

| Command | What it does |
|---|---|
| `ingest` | Extracts metadata and registers videos in `video_registry.json` |
| `chunk` | Divides registered videos into fixed-length chunks (`chunk_registry.json`) |
| `frames` | Extracts sampled frames from each chunk to `data/processed/frames/` |
| `caption` | Captions each frame using BLIP and saves to `caption_registry.json` |
| `merge` | Combines all registries into `merged_registry.json` |
| `embed` | Embeds chunks into the ChromaDB vector store |

```bash
python -m src.ingestion_pipeline.cli ingest --videos data/raw/new_dive.MP4
python -m src.ingestion_pipeline.cli chunk --duration 20
python -m src.ingestion_pipeline.cli frames --interval 3.33
python -m src.ingestion_pipeline.cli caption
python -m src.ingestion_pipeline.cli merge
python -m src.ingestion_pipeline.cli embed
```

### Options

| Flag | Applies to | Default | Description |
|---|---|---|---|
| `--videos PATH [PATH ...]` | `run`, `ingest` | All MP4s in `data/raw/` | Specific video file(s) to process |
| `--duration S` | `run`, `chunk` | `20.0` | Chunk length in seconds |
| `--interval S` | `run`, `frames` | `3.33` | Frame sampling interval in seconds |
| `--output-dir DIR` | `run`, `frames` | `data/processed/frames/` | Root directory for extracted frames |

### Adding new videos

The `embed` step uses `update()` under the hood, so only newly added chunks are embedded — already-indexed chunks are skipped.

```bash
# Drop new files into data/raw/, then:
python -m src.ingestion_pipeline.cli ingest
python -m src.ingestion_pipeline.cli chunk
python -m src.ingestion_pipeline.cli frames
python -m src.ingestion_pipeline.cli caption
python -m src.ingestion_pipeline.cli merge
python -m src.ingestion_pipeline.cli embed
```

### Output files

| File | Contents |
|---|---|
| `data/processed/video_registry.json` | Video metadata (resolution, GPS, timestamps) |
| `data/processed/chunk_registry.json` | Chunk time boundaries and frame ranges |
| `data/processed/caption_registry.json` | Per-frame captions keyed by chunk |
| `data/processed/merged_registry.json` | Combined view used for embedding |
| `data/processed/frames/` | Extracted JPEG frames organised by video and chunk |
| `data/processed/vector_store/` | ChromaDB persistent storage |
