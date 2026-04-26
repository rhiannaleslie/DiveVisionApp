import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.constants import MERGED_REGISTRY_JSON, VECTOR_STORE_DIR

DEFAULT_COLLECTION = "dive_chunks"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ChunkVectorStore:
    """ChromaDB-backed semantic index over dive video chunks.

    Each chunk is indexed as a single document combining its captions and key
    metadata (video name, timestamps, GPS location). Metadata fields are stored
    separately so results can be filtered by location, date, video, etc.

    Typical usage::

        store = ChunkVectorStore()
        store.build()
        results = store.query("turtle swimming near coral reef", n_results=5)
    """

    def __init__(
        self,
        persist_dir: Path = VECTOR_STORE_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """Configure the vector store.

        Args:
            persist_dir: Directory where ChromaDB persists its data on disk.
            collection_name: Name of the ChromaDB collection to use.
            embedding_model: Sentence-Transformers model name used to embed documents.
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._collection = None

    def _connect(self) -> None:
        """Lazily initialise the ChromaDB client and collection."""
        if self._collection is not None:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        ef = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ef,
        )

    def build(self, merged_registry_path: Path = MERGED_REGISTRY_JSON) -> int:
        """Index all chunks from the merged registry into ChromaDB.

        Reads the merged registry JSON (which combines chunk, caption, and video
        metadata), builds a text document and metadata dict for each chunk, then
        upserts them into the collection. Upsert is used so re-running is safe
        and idempotent — existing chunks are updated rather than duplicated.

        Args:
            merged_registry_path: Path to the merged_registry.json file.

        Returns:
            Number of chunks indexed.

        Raises:
            FileNotFoundError: If the merged registry does not exist.
        """
        if not merged_registry_path.exists():
            raise FileNotFoundError(
                f"Merged registry not found: {merged_registry_path}. "
                "Run merge_registries() first."
            )

        self._connect()

        merged = json.loads(merged_registry_path.read_text())

        ids = []
        documents = []
        metadatas = []

        for chunk_id, chunk in merged.items():
            ids.append(chunk_id)
            documents.append(self._make_document(chunk))
            metadatas.append(self._make_metadata(chunk))

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Indexed {len(ids)} chunks into '{self.collection_name}'")
        return len(ids)

    def update(self, merged_registry_path: Path = MERGED_REGISTRY_JSON) -> int:
        """Index only chunks that are not yet present in the collection.

        Intended to be called after new videos have been ingested, chunked,
        captioned, and merged. Already-indexed chunks are skipped entirely, so
        only the new embeddings are computed and stored.

        Args:
            merged_registry_path: Path to the merged_registry.json file.

        Returns:
            Number of new chunks indexed (0 if already up to date).

        Raises:
            FileNotFoundError: If the merged registry does not exist.
        """
        if not merged_registry_path.exists():
            raise FileNotFoundError(
                f"Merged registry not found: {merged_registry_path}. "
                "Run merge_registries() first."
            )

        self._connect()

        merged = json.loads(merged_registry_path.read_text())

        existing_ids = set(self._collection.get(include=[])["ids"])
        new_chunks = {
            chunk_id: chunk
            for chunk_id, chunk in merged.items()
            if chunk_id not in existing_ids
        }

        if not new_chunks:
            print("Vector store is already up to date.")
            return 0

        ids = list(new_chunks.keys())
        documents = [self._make_document(chunk) for chunk in new_chunks.values()]
        metadatas = [self._make_metadata(chunk) for chunk in new_chunks.values()]

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Indexed {len(ids)} new chunks into '{self.collection_name}'")
        return len(ids)

    def remove_video(self, video_id: str) -> int:
        """Remove all indexed chunks belonging to a video.

        Useful when a video is deleted or re-ingested from scratch. After
        removal, call update() to re-index the video's new chunks.

        Args:
            video_id: The UUID of the video whose chunks should be removed.

        Returns:
            Number of chunks removed.
        """
        self._connect()
        results = self._collection.get(where={"video_id": video_id}, include=[])
        ids_to_remove = results["ids"]
        if ids_to_remove:
            self._collection.delete(ids=ids_to_remove)
            print(f"Removed {len(ids_to_remove)} chunks for video_id={video_id}")
        else:
            print(f"No chunks found for video_id={video_id}")
        return len(ids_to_remove)

    def count(self) -> int:
        """Return the total number of chunks currently in the index.

        Returns:
            Integer count of indexed chunks.
        """
        self._connect()
        return self._collection.count()

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Run a semantic search against the indexed chunks.

        Args:
            query_text: Natural-language search query (e.g. "shark near coral reef").
            n_results: Number of closest matches to return.

        Returns:
            List of result dicts ordered by relevance (closest first), each with:
                - chunk_id: str
                - distance: float (lower = more similar)
                - document: str (the text that was embedded)
                - metadata: dict (all stored metadata fields)
        """
        self._connect()

        raw = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )

        results = []
        for i in range(len(raw["ids"][0])):
            results.append({
                "chunk_id": raw["ids"][0][i],
                "distance": raw["distances"][0][i],
                "document": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
            })

        return results

    def _make_document(self, chunk: dict) -> str:
        """Build the embeddable text document for a single chunk.

        Combines captions with structured context (video name, time range, GPS
        location, recording date) so the embedding captures both visual content
        and situational metadata.

        Args:
            chunk: A merged registry entry with captions and video sub-dict.

        Returns:
            Plain-text document string ready for embedding.
        """
        video = chunk.get("video", {})
        captions = chunk.get("captions", {})

        parts = []

        filename = video.get("filename", "unknown")
        parts.append(
            f"Video: {filename}, chunk {chunk['chunk_index']}, "
            f"{chunk['start_s']}s to {chunk['end_s']}s"
        )

        created_on = video.get("created_on")
        if created_on:
            parts.append(f"Recorded: {str(created_on)[:10]}")

        lat = video.get("gps_lat_min")
        lon = video.get("gps_lon_min")
        if lat is not None and lon is not None:
            parts.append(f"Location: {lat:.5f}, {lon:.5f}")

        if captions:
            parts.append("Captions: " + ". ".join(captions.values()))

        return "\n".join(parts)

    def _make_metadata(self, chunk: dict) -> dict:
        """Extract a ChromaDB-compatible metadata dict from a merged chunk record.

        ChromaDB only accepts str, int, float, and bool metadata values.
        None values are converted to empty strings (for text fields) or 0 (for
        numeric fields) to satisfy this constraint.

        Args:
            chunk: A merged registry entry with captions and video sub-dict.

        Returns:
            Flat dict of metadata suitable for ChromaDB storage and filtering.
        """
        video = chunk.get("video", {})

        def _str(v) -> str:
            return str(v) if v is not None else ""

        def _float(v) -> float:
            return float(v) if v is not None else 0.0

        def _int(v) -> int:
            return int(v) if v is not None else 0

        return {
            "chunk_id": chunk["chunk_id"],
            "video_id": chunk["video_id"],
            "chunk_index": chunk["chunk_index"],
            "start_s": chunk["start_s"],
            "end_s": chunk["end_s"],
            "duration_s": chunk["duration_s"],
            "video_filename": _str(video.get("filename")),
            "created_on": _str(video.get("created_on")),
            "gps_lat_min": _float(video.get("gps_lat_min")),
            "gps_lat_max": _float(video.get("gps_lat_max")),
            "gps_lon_min": _float(video.get("gps_lon_min")),
            "gps_lon_max": _float(video.get("gps_lon_max")),
            "gps_start_ts": _str(video.get("gps_start_ts")),
            "gps_end_ts": _str(video.get("gps_end_ts")),
            "iso_min": _int(video.get("iso_min")),
            "iso_max": _int(video.get("iso_max")),
            "fps": _float(video.get("fps")),
            "width": _int(video.get("width")),
            "height": _int(video.get("height")),
        }
