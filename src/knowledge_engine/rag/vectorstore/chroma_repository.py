"""
ChromaDB vector store repository for the GuardianRAG system (Enterprise Edition)
 
Implements the Repository pattern over ChromaDB.
 
Responsibilities:
  - Initialise a persistent ChromaDB collection
  - Add chunks with their embedding vectors and rich metadata
  - Query by similarity vector (with optional score threshold + metadata filters)
  - Delete individual records or entire collections
  - Expose collection statistics
 
Persistence:
  Vectors are stored on disk under Settings.CHROMA_PERSIST_DIR
  (default: ./data/vectorstore).  Survives process restarts.
 
Metadata stored per chunk:
  - document_id    string    unique document identifier
  - chunk_index    int       position of chunk within document
  - source_file    string    original PDF / file path
  - page_number    int       estimated page (from Week 2 metadata)
  - section        string    detected section heading (or "")
  - char_start     int       character offset in cleaned text
  - char_end       int       character offset end in cleaned text
  - word_count     int       number of words in chunk
  - quality_score  float     document quality score from Week 2
  - chunker_name   string    which chunker produced this chunk
 
Usage:
    repo = get_chroma_repository()
    repo.add_chunks(chunks_with_embeddings)
    results = repo.query(embedding_vector, top_k=5)
"""
from __future__ import annotations
 
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
 
import chromadb
from chromadb.config import Settings as ChromaSettings
 
from knowledge_engine.core.config import get_settings
from knowledge_engine.core.exceptions import APIError
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
# ─── Exceptions ──────────────────────────────────────────────────────────────
 
 
class VectorStoreError(APIError):
    """Raised when a ChromaDB operation fails."""
 
 
# ─── Data Contracts ──────────────────────────────────────────────────────────
 
 
@dataclass
class ChunkRecord:
    """
    A chunk ready to be stored in the vector database.
 
    This is the contract between the IndexManager and the ChromaRepository:
    the caller must supply both the embedding and the full set of metadata
    fields that the retrieval layer expects.
 
    Fields:
        chunk_id:      Stable unique ID (SHA-256 based from Week 2 metadata
                       builder, or caller-generated UUID).
        text:          Original chunk text (stored as ChromaDB document).
        embedding:     Float vector from EmbeddingProvider.
        document_id:   Logical document identifier (e.g. filename stem).
        chunk_index:   Zero-based position within the document.
        source_file:   Path to the original source file.
        page_number:   Estimated page number (-1 if unknown).
        section:       Section heading detected by Week 2 semantic chunker ("" if none).
        char_start:    Character start offset in the cleaned document text.
        char_end:      Character end offset in the cleaned document text.
        word_count:    Number of words in the chunk.
        quality_score: Document quality score (0.0–1.0) from Week 2 validator.
        chunker_name:  Name of the chunker that produced this chunk.
        extra_metadata: Any additional key/value pairs to persist.
    """
 
    chunk_id: str
    text: str
    embedding: list[float]
    document_id: str
    chunk_index: int
    source_file: str
    page_number: int = -1
    section: str = ""
    char_start: int = 0
    char_end: int = 0
    word_count: int = 0
    quality_score: float = 0.0
    chunker_name: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)
 
    def to_chroma_metadata(self) -> dict[str, Any]:
        """
        Return a flat metadata dict safe for ChromaDB storage.
 
        ChromaDB metadata values must be str, int, float, or bool.
        All fields are coerced accordingly.
        """
        meta: dict[str, Any] = {
            "document_id": str(self.document_id),
            "chunk_index": int(self.chunk_index),
            "source_file": str(self.source_file),
            "page_number": int(self.page_number),
            "section": str(self.section),
            "char_start": int(self.char_start),
            "char_end": int(self.char_end),
            "word_count": int(self.word_count),
            "quality_score": float(self.quality_score),
            "chunker_name": str(self.chunker_name),
        }
        # Merge extra metadata — coerce values to safe types
        for k, v in self.extra_metadata.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = str(v)
        return meta
 
 
@dataclass
class QueryResult:
    """
    A single result returned from a vector similarity query.
 
    Fields:
        chunk_id:    ChromaDB document ID.
        text:        Original chunk text.
        score:       Cosine similarity score (0.0–1.0, higher = more similar).
                     ChromaDB returns *distances* (lower = closer); the
                     repository converts: score = 1 - distance.
        metadata:    All metadata fields as stored in ChromaDB.
    """
 
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]
 
    def get(self, key: str, default: Any = None) -> Any:
        """Convenience accessor for metadata fields."""
        return self.metadata.get(key, default)
 
    def __repr__(self) -> str:
        doc = self.metadata.get("document_id", "?")
        idx = self.metadata.get("chunk_index", "?")
        return (
            f"QueryResult(score={self.score:.4f}, "
            f"doc={doc!r}, chunk={idx}, "
            f"text={self.text[:50]!r}{'...' if len(self.text) > 50 else ''})"
        )
 
 
# ─── Repository ──────────────────────────────────────────────────────────────
 
 
class ChromaRepository:
    """
    Repository layer over a single ChromaDB collection.
 
    One ChromaRepository instance manages one collection (default name:
    "guardianrag_chunks").  Use different collection names for multi-tenant
    or multi-document-set deployments.
 
    Enterprise features:
        - Persist directory read from Settings.CHROMA_PERSIST_DIR
        - Score threshold filtering (converts distance → similarity)
        - Metadata filtering via ChromaDB `where` clauses
        - Idempotent add (upsert semantics — duplicate chunk_ids are updated)
        - Batch add with configurable chunk size
        - Collection statistics for monitoring
    """
 
    _DEFAULT_COLLECTION = "guardianrag_chunks"
 
    def __init__(
        self,
        collection_name: str = _DEFAULT_COLLECTION,
        persist_dir: str | None = None,
    ) -> None:
        """
        Initialise the repository.
 
        Args:
            collection_name: ChromaDB collection to use / create.
            persist_dir:     Override the Settings.CHROMA_PERSIST_DIR path.
        """
        settings = get_settings()
        self._persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._collection_name = collection_name
 
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
 
        logger.info(
            "ChromaRepository ready: collection=%r, persist_dir=%r, existing_docs=%d",
            collection_name,
            self._persist_dir,
            self._collection.count(),
        )
 
    # ── Write operations ─────────────────────────────────────────────────
 
    def add_chunks(
        self,
        chunks: list[ChunkRecord],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert a list of ChunkRecords into the collection.
 
        Duplicate chunk_ids are updated (upsert), so this is safe to call
        multiple times for the same document (idempotent).
 
        Args:
            chunks:     ChunkRecords to store (must have non-empty embedding).
            batch_size: How many records to upsert per ChromaDB call.
 
        Returns:
            Number of chunks successfully stored.
 
        Raises:
            VectorStoreError: If ChromaDB raises an unexpected exception.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list — nothing to do")
            return 0
 
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            try:
                self._upsert_batch(batch)
                total += len(batch)
                logger.debug(
                    "Upserted batch %d-%d (%d chunks) into collection %r",
                    i,
                    i + len(batch) - 1,
                    len(batch),
                    self._collection_name,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to upsert batch [{i}:{i+len(batch)}]: {exc}"
                ) from exc
 
        logger.info(
            "add_chunks complete: %d chunks stored in %r",
            total,
            self._collection_name,
        )
        return total
 
    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.
 
        Args:
            document_id: The document_id metadata field to match.
 
        Returns:
            Number of chunks deleted (approximated — ChromaDB does not return count).
        """
        try:
            before = self._collection.count()
            self._collection.delete(where={"document_id": {"$eq": document_id}})
            after = self._collection.count()
            deleted = before - after
            logger.info(
                "Deleted document %r from collection %r (%d chunks removed)",
                document_id,
                self._collection_name,
                deleted,
            )
            return deleted
        except Exception as exc:
            raise VectorStoreError(
                f"delete_document failed for {document_id!r}: {exc}"
            ) from exc
 
    def clear_collection(self) -> None:
        """
        Delete all records from the collection (destructive — use with care).
        """
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.warning("Collection %r cleared.", self._collection_name)
        except Exception as exc:
            raise VectorStoreError(f"clear_collection failed: {exc}") from exc
 
    # ── Read / Query operations ──────────────────────────────────────────
 
    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        where: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """
        Retrieve the most similar chunks for a query embedding.
 
        Args:
            embedding:       Query vector (must match the stored dimension).
            top_k:           Maximum number of results to return.
            score_threshold: Minimum similarity score (0.0–1.0).  Results
                             below this threshold are filtered out.
            where:           ChromaDB metadata filter expression.
                             Example: {"document_id": {"$eq": "policy_v3"}}
                             See: https://docs.trychroma.com/usage-guide#filtering-by-metadata
 
        Returns:
            List of QueryResult objects, sorted by descending similarity score,
            filtered by score_threshold.
 
        Raises:
            VectorStoreError: On ChromaDB query failure.
        """
        if self._collection.count() == 0:
            logger.warning("query called on empty collection %r", self._collection_name)
            return []
 
        try:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": min(top_k, self._collection.count()),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where
 
            raw = self._collection.query(**query_kwargs)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc
 
        results = self._parse_query_response(raw, score_threshold)
 
        logger.debug(
            "query returned %d results (top_k=%d, threshold=%.2f, collection=%r)",
            len(results),
            top_k,
            score_threshold,
            self._collection_name,
        )
        return results
 
    def get_by_ids(self, chunk_ids: list[str]) -> list[QueryResult]:
        """
        Fetch specific chunks by their IDs (no similarity ranking).
 
        Args:
            chunk_ids: List of chunk_id values to retrieve.
 
        Returns:
            QueryResult list (score=1.0 for direct lookup — not a similarity score).
        """
        try:
            raw = self._collection.get(
                ids=chunk_ids,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(f"get_by_ids failed: {exc}") from exc
 
        results = []
        for cid, doc, meta in zip(
            raw.get("ids", []),
            raw.get("documents", []),
            raw.get("metadatas", []),
        ):
            results.append(
                QueryResult(
                    chunk_id=cid,
                    text=doc or "",
                    score=1.0,
                    metadata=meta or {},
                )
            )
        return results
 
    # ── Statistics ───────────────────────────────────────────────────────
 
    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self._collection.count()
 
    def stats(self) -> dict[str, Any]:
        """Return a summary dict for monitoring / health checks."""
        return {
            "collection": self._collection_name,
            "persist_dir": self._persist_dir,
            "total_chunks": self._collection.count(),
        }
 
    # ── Private helpers ──────────────────────────────────────────────────
 
    def _upsert_batch(self, batch: list[ChunkRecord]) -> None:
        self._collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=[c.embedding for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.to_chroma_metadata() for c in batch],
        )
 
    @staticmethod
    def _parse_query_response(
        raw: dict[str, Any],
        score_threshold: float,
    ) -> list[QueryResult]:
        """
        Convert ChromaDB raw query response to QueryResult objects.
 
        ChromaDB uses *cosine distance* (lower = more similar).
        We convert: similarity_score = 1 - distance.
        """
        ids_list = raw.get("ids", [[]])[0]
        docs_list = raw.get("documents", [[]])[0]
        metas_list = raw.get("metadatas", [[]])[0]
        dists_list = raw.get("distances", [[]])[0]
 
        results: list[QueryResult] = []
        for cid, doc, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            score = max(0.0, 1.0 - dist)  # cosine distance → similarity
            if score < score_threshold:
                continue
            results.append(
                QueryResult(
                    chunk_id=cid,
                    text=doc or "",
                    score=round(score, 6),
                    metadata=meta or {},
                )
            )
 
        # Already sorted by ChromaDB (nearest first), but re-sort after filtering
        results.sort(key=lambda r: r.score, reverse=True)
        return results
 
 
# ─── Factory ─────────────────────────────────────────────────────────────────
 
 
@lru_cache(maxsize=4)
def get_chroma_repository(
    collection_name: str = ChromaRepository._DEFAULT_COLLECTION,
    persist_dir: str | None = None,
) -> ChromaRepository:
    """
    Return a cached ChromaRepository.
 
    Follows the same factory pattern as get_settings() / get_token_calculator().
    The cache key includes collection_name so multi-collection scenarios are
    supported without creating duplicate clients.
 
    Args:
        collection_name: ChromaDB collection name.
        persist_dir:     Override persist directory (tests use tmp dirs).
 
    Returns:
        ChromaRepository singleton per (collection_name, persist_dir) pair.
    """
    return ChromaRepository(collection_name=collection_name, persist_dir=persist_dir)
 
 
def make_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Generate a stable chunk ID from document_id + chunk_index.
 
    Prefer the SHA-256 chunk_id from Week 2's ChunkMetadataBuilder when
    available.  Use this helper when that ID is not present.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}::{chunk_index}"))
