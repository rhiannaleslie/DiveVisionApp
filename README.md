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

Or run commands directly without activating:

```bash
uv run python src/ingestion.py
```
