# ![Status](https://img.shields.io/badge/status-in%20development-yellow) DiveVisionApp

A retrieval-augmented generation (RAG) app for searching a personal scuba diving video library by visual features.

## Overview

DiveVisionApp processes GoPro dive footage into a searchable library. Videos are ingested, chunked into scenes, and frames are extracted and embedded so that natural-language or feature-based queries can retrieve relevant moments — e.g. "find clips with a turtle" or "show me low-visibility drift dives".

## Rough aims

- Ingest raw dive footage and build a structured video registry
- Chunk videos into meaningful scenes and extract representative frames
- Embed frames using a vision model to enable semantic search
- Expose a RAG interface for querying the library by visual features, dive conditions, or marine life

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then:

```bash
uv sync --dev
```

This creates a virtual environment at `.venv/` and installs all dependencies. To activate it:

```bash
source .venv/bin/activate
```

Or run commands directly without activating:

```bash
uv run python src/ingestion.py
```

## Example

[notebooks/pipeline_test.ipynb](notebooks/pipeline_test.ipynb) contains a full end-to-end pipeline run across 3 dive videos — ingestion, chunking, frame extraction, and embedding in one place.
