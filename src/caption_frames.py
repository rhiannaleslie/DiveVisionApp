import json

import torch
from pathlib import Path
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.chunk_video import CHUNK_REGISTRY_JSON
from src.constants import FRAMES_DIR, PROCESSED_DIR
from src.extract_frames import chunk_output_dir
from src.ingestion import load_registry

CAPTION_REGISTRY_JSON = PROCESSED_DIR / "caption_registry.json"
MODEL_ID = "Salesforce/blip-image-captioning-large"
MAX_NEW_TOKENS = 50


def load_model() -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    return processor, model


def caption_image(
    image_path: Path,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
) -> str:
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    return processor.decode(output[0], skip_special_tokens=True)


def caption_all_chunks(
    chunk_ids: list[str] | None = None,
) -> dict[str, dict]:
    """Caption all frames for all chunks and save to caption_registry.json.

    Args:
        chunk_ids: Optional list of chunk_ids to restrict processing to.

    Returns:
        The full caption registry dict (chunk_id -> record with captions).
    """
    video_registry = load_registry()
    chunk_registry: dict = json.loads(CHUNK_REGISTRY_JSON.read_text())

    if chunk_ids is not None:
        chunk_registry = {cid: c for cid, c in chunk_registry.items() if cid in chunk_ids}

    existing: dict = {}
    if CAPTION_REGISTRY_JSON.exists():
        existing = json.loads(CAPTION_REGISTRY_JSON.read_text())

    processor, model = load_model()

    for chunk_id, chunk in chunk_registry.items():
        video_record = video_registry[chunk["video_id"]]
        frames_dir = chunk_output_dir(chunk, video_record["filename"], FRAMES_DIR)

        if not frames_dir.exists():
            print(f"Skipping {chunk_id}: frames dir not found ({frames_dir})")
            continue

        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            print(f"Skipping {chunk_id}: no frames found")
            continue

        captions: dict[str, str] = {}
        for frame_path in frame_paths:
            caption = caption_image(frame_path, processor, model)
            captions[frame_path.name] = caption
            print(f"  {frame_path.name}: {caption}")

        existing[chunk_id] = {
            "chunk_id": chunk_id,
            "video_id": chunk["video_id"],
            "chunk_index": chunk["chunk_index"],
            "captions": captions,
        }
        print(f"Captioned chunk {chunk_id} ({len(captions)} frames)")

    CAPTION_REGISTRY_JSON.write_text(json.dumps(existing, indent=2))
    print(f"\nSaved caption registry to {CAPTION_REGISTRY_JSON}")
    return existing


