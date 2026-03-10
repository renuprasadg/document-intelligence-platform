"""
Index Manager for the GuardianRAG system (Enterprise Edition)
 
Implements the indexing pipeline:
 
    JSONL chunk files (data/chunks/)
        → read chunks
        → normalise text for embedding
        → generate embeddings via EmbeddingProvider
        → store ChunkRecords in ChromaRepository
 
Responsibilities:
  - Discover and read JSONL files produced by Week 2 chunk_documents_cli.py
  - Map Week 2 chunk metadata fields → ChunkRecord fields
  - Batch-embed with configurable batch size
  - Store in ChromaDB (upsert — safe to re-index)
  - Report indexing statistics
  - Skip already-indexed documents (optional, via document presence check)
 
JSONL schema expected (from Week 2 chunk_documents_cli.py output):
    {
        "text": "...",
        "metadata": {
            "chunk_id":      "sha256-based-id",
            "source_path":   "path/to/original.pdf",
            "document_title":"Policy v3",
            "chunk_index":   0,
            "total_chunks":  42,
            "char_start":    0,
            "char_end":      512,
            "page_estimate": 1,
            "word_count":    87,
            "quality_score": 0.91,
            "quality_level": "HIGH",
            "chunker_name":  "SemanticChunker"
        }
    }
 
Usage:
    manager = get_index_manager()
    stats   = manager.index_directory("data/chunks")
    print(stats)
"""
from __future__ import annotations
 
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
 
from knowledge_engine.core.exceptions import DocumentProcessingError
from knowledge_engine.core.logging_config import get_logger
from knowledge_engine.rag.embeddings.embedding_provider import EmbeddingProvider
from knowledge_engine.rag.vectorstore.chroma_repository import (
    ChunkRecord,
    ChromaRepository,
    VectorStoreError,
    make_chunk_id,
)
 
logger = get_logger(__name__)
 
 
# ─── Result / Statistics Dataclasses ────────────────────────────────────────
 
 
@dataclass
class IndexStats:
    """
    Statistics for a completed indexing run.
 
    Fields:
        files_processed:    Number of JSONL files consumed.
        chunks_read:        Total chunks read from JSONL files.
        chunks_embedded:    Total chunks successfully embedded.
        chunks_stored:      Total chunks stored in ChromaDB.
        chunks_skipped:     Chunks skipped (e.g. empty text, already indexed).
        errors:             List of error messages for failed chunks / files.
        elapsed_seconds:    Wall-clock seconds for the full run.
    """
 
    files_processed: int = 0
    chunks_read: int = 0
    chunks_embedded: int = 0
    chunks_stored: int = 0
    chunks_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
 
    def __str__(self) -> str:
        return (
            f"IndexStats("
            f"files={self.files_processed}, "
            f"read={self.chunks_read}, "
            f"embedded={self.chunks_embedded}, "
            f"stored={self.chunks_stored}, "
            f"skipped={self.chunks_skipped}, "
            f"errors={len(self.errors)}, "
            f"elapsed={self.elapsed_seconds:.1f}s)"
        )
 
    @property
    def success(self) -> bool:
        """True if at least one chunk was stored and no errors occurred."""
        return self.chunks_stored > 0 and len(self.errors) == 0
 
 
@dataclass
class IndexManagerConfig:
    """
    Runtime configuration for IndexManager.
 
    Fields:
        chunks_dir:      Default directory to scan for JSONL files.
        embedding_batch_size: Number of chunks to embed per API call.
        store_batch_size:     Number of ChunkRecords to upsert per ChromaDB call.
        skip_empty_text:     Skip chunks with empty/whitespace-only text.
        file_glob:           Glob pattern for JSONL discovery (default *.jsonl).
    """
 
    chunks_dir: str = "data/chunks"
    embedding_batch_size: int = 50
    store_batch_size: int = 100
    skip_empty_text: bool = True
    file_glob: str = "*.jsonl"
 
 
# ─── IndexManager ────────────────────────────────────────────────────────────
 
 
class IndexManager:
    """
    Orchestrates the full chunks → embeddings → vector store pipeline.
 
    Dependency-injected: accepts any EmbeddingProvider and ChromaRepository,
    making it easy to test with stubs without hitting the OpenAI API.
 
    Usage:
        # Production
        manager = IndexManager(
            embedding_provider=get_openai_embedding_provider(),
            repository=get_chroma_repository(),
        )
        stats = manager.index_directory("data/chunks")
 
        # Testing (inject stubs)
        manager = IndexManager(
            embedding_provider=FakeEmbeddingProvider(),
            repository=ChromaRepository(persist_dir=tmp_dir),
        )
    """
 
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        repository: ChromaRepository,
        config: IndexManagerConfig | None = None,
    ) -> None:
        self._provider = embedding_provider
        self._repo = repository
        self._config = config or IndexManagerConfig()
 
        logger.info(
            "IndexManager initialised: provider=%s, collection=%r",
            self._provider.model_name,
            self._repo.stats().get("collection"),
        )
 
    # ── Public API ───────────────────────────────────────────────────────
 
    def index_directory(
        self,
        directory: str | None = None,
        glob: str | None = None,
    ) -> IndexStats:
        """
        Read all JSONL files in a directory and index them into ChromaDB.
 
        Args:
            directory: Path to directory containing *.jsonl chunk files.
                       Defaults to IndexManagerConfig.chunks_dir.
            glob:      Override the file glob pattern.
 
        Returns:
            IndexStats summarising the run.
        """
        dir_path = Path(directory or self._config.chunks_dir)
        if not dir_path.exists():
            raise DocumentProcessingError(
                f"Chunks directory not found: {dir_path}"
            )
 
        pattern = glob or self._config.file_glob
        jsonl_files = sorted(dir_path.glob(pattern))
 
        if not jsonl_files:
            logger.warning("No JSONL files found in %s (glob=%r)", dir_path, pattern)
            return IndexStats()
 
        logger.info(
            "Indexing %d JSONL file(s) from %s",
            len(jsonl_files),
            dir_path,
        )
 
        stats = IndexStats()
        start = time.monotonic()
 
        for jsonl_path in jsonl_files:
            self._index_file(jsonl_path, stats)
 
        stats.elapsed_seconds = time.monotonic() - start
        logger.info("Indexing complete: %s", stats)
        return stats
 
    def index_file(self, filepath: str | Path) -> IndexStats:
        """
        Index a single JSONL file.
 
        Args:
            filepath: Path to a .jsonl file produced by Week 2 CLI.
 
        Returns:
            IndexStats for this file.
        """
        stats = IndexStats()
        start = time.monotonic()
        self._index_file(Path(filepath), stats)
        stats.elapsed_seconds = time.monotonic() - start
        return stats
 
    def index_chunks(self, raw_chunks: list[dict[str, Any]]) -> IndexStats:
        """
        Index an in-memory list of chunk dicts (same schema as JSONL lines).
 
        Useful for programmatic use without writing files to disk.
 
        Args:
            raw_chunks: List of {"text": "...", "metadata": {...}} dicts.
 
        Returns:
            IndexStats for this batch.
        """
        stats = IndexStats()
        start = time.monotonic()
        stats.chunks_read = len(raw_chunks)
        self._embed_and_store(raw_chunks, stats)
        stats.elapsed_seconds = time.monotonic() - start
        return stats
 
    # ── Internal pipeline ────────────────────────────────────────────────
 
    def _index_file(self, path: Path, stats: IndexStats) -> None:
        """Read one JSONL file and pass its chunks through the pipeline."""
        logger.info("Reading JSONL file: %s", path)
        try:
            raw_chunks = list(_iter_jsonl(path))
        except Exception as exc:
            msg = f"Failed to read {path}: {exc}"
            logger.error(msg)
            stats.errors.append(msg)
            return
 
        stats.files_processed += 1
        stats.chunks_read += len(raw_chunks)
        self._embed_and_store(raw_chunks, stats)
 
    def _embed_and_store(
        self,
        raw_chunks: list[dict[str, Any]],
        stats: IndexStats,
    ) -> None:
        """Embed raw chunk dicts and upsert them into the repository."""
        # Filter out empty-text chunks
        valid: list[dict[str, Any]] = []
        for raw in raw_chunks:
            text = raw.get("text", "")
            if self._config.skip_empty_text and not text.strip():
                logger.debug("Skipping empty-text chunk")
                stats.chunks_skipped += 1
            else:
                valid.append(raw)
 
        if not valid:
            return
 
        # Embed in batches
        batch_size = self._config.embedding_batch_size
        for i in range(0, len(valid), batch_size):
            batch = valid[i : i + batch_size]
            texts = [r["text"] for r in batch]
 
            try:
                batch_result = self._provider.embed_batch(texts)
                stats.chunks_embedded += len(batch_result.results)
            except Exception as exc:
                msg = f"Embedding failed for batch [{i}:{i+len(batch)}]: {exc}"
                logger.error(msg)
                stats.errors.append(msg)
                continue
 
            # Build ChunkRecords
            records: list[ChunkRecord] = []
            for raw_chunk, emb_result in zip(batch, batch_result.results):
                try:
                    record = _build_chunk_record(raw_chunk, emb_result.embedding)
                    records.append(record)
                except Exception as exc:
                    msg = f"Failed to build ChunkRecord: {exc}"
                    logger.warning(msg)
                    stats.errors.append(msg)
                    stats.chunks_skipped += 1
 
            # Store in ChromaDB
            if records:
                try:
                    stored = self._repo.add_chunks(
                        records, batch_size=self._config.store_batch_size
                    )
                    stats.chunks_stored += stored
                except VectorStoreError as exc:
                    msg = f"Failed to store batch in ChromaDB: {exc}"
                    logger.error(msg)
                    stats.errors.append(msg)
 
 
# ─── JSONL reader ────────────────────────────────────────────────────────────
 
 
def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed dicts from a JSONL file, skipping blank/comment lines."""
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at %s line %d: %s", path, lineno, exc)
 
 
# ─── ChunkRecord builder ─────────────────────────────────────────────────────
 
 
def _build_chunk_record(
    raw: dict[str, Any],
    embedding: list[float],
) -> ChunkRecord:
    """
    Map a raw JSONL chunk dict to a ChunkRecord.
 
    Handles both the canonical Week 2 schema and loose variants gracefully.
    """
    text: str = raw.get("text", "")
    meta: dict[str, Any] = raw.get("metadata", {})
 
    # Derive document_id from source_path stem (e.g. "policy_v3.pdf" → "policy_v3")
    source_path: str = meta.get("source_path", meta.get("source_file", ""))
    document_id: str = meta.get(
        "document_id",
        Path(source_path).stem if source_path else "unknown",
    )
 
    # Use Week 2 chunk_id if available; fall back to deterministic UUID
    chunk_id: str = meta.get(
        "chunk_id",
        make_chunk_id(document_id, int(meta.get("chunk_index", 0))),
    )
 
    return ChunkRecord(
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
        document_id=document_id,
        chunk_index=int(meta.get("chunk_index", 0)),
        source_file=source_path,
        page_number=int(meta.get("page_estimate", meta.get("page_number", -1))),
        section=str(meta.get("section", "")),
        char_start=int(meta.get("char_start", 0)),
        char_end=int(meta.get("char_end", 0)),
        word_count=int(meta.get("word_count", 0)),
        quality_score=float(meta.get("quality_score", 0.0)),
        chunker_name=str(meta.get("chunker_name", "")),
        extra_metadata={
            "document_title": meta.get("document_title", ""),
            "total_chunks": meta.get("total_chunks", -1),
            "quality_level": meta.get("quality_level", ""),
        },
    )
 
 
# ─── Factory ─────────────────────────────────────────────────────────────────
 
 
@lru_cache(maxsize=1)
def get_index_manager() -> "IndexManager":
    """
    Return a cached IndexManager using the default OpenAI provider and
    ChromaDB repository configured from Settings.
 
    Import inside function to avoid circular imports at module load time.
    """
    from knowledge_engine.adapters.openai.openai_embeddings import (
        get_openai_embedding_provider,
    )
    from knowledge_engine.rag.vectorstore.chroma_repository import get_chroma_repository
 
    return IndexManager(
        embedding_provider=get_openai_embedding_provider(),
        repository=get_chroma_repository(),
    )
