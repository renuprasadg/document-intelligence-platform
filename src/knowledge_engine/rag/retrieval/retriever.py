"""
Retriever for the GuardianRAG system (Enterprise Edition)
 
The main entry point for the RAG retrieval pipeline.
 
Full pipeline (per query):
 
    user query string
        → embed query (EmbeddingProvider)
        → similarity search + metadata filtering (ChromaRepository)
        → score threshold filtering
        → max-chunks-per-document capping
        → optional reranking (Reranker)
        → RetrievalResult list
 
Responsibilities:
  - Accept a natural-language query string
  - Delegate embedding to EmbeddingProvider
  - Delegate vector search to ChromaRepository (with metadata where clause)
  - Apply score_threshold and per-document chunk limits
  - Delegate optional reranking to Reranker
  - Return typed, structured results
 
Design notes:
  - Retriever is dependency-injected: tests can swap all three collaborators.
  - Settings defaults (TOP_K, SIMILARITY_THRESHOLD) are used when callers
    do not specify overrides, matching the Week 1 config.py fields.
  - All filter logic is delegated to retrieval_filters.py — the Retriever
    does NOT build where dicts directly.
 
Usage:
    retriever = get_retriever()
    results   = retriever.search("What is the flood damage deductible?")
 
    for r in results:
        print(r.text)
        print(f"  score={r.score:.3f}  doc={r.document_id}  page={r.page_number}")
"""
from __future__ import annotations
 
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
 
from knowledge_engine.core.config import get_settings
from knowledge_engine.core.exceptions import APIError
from knowledge_engine.core.logging_config import get_logger
from knowledge_engine.rag.embeddings.embedding_provider import (
    EmbeddingProvider,
    EmbeddingError,
)
from knowledge_engine.rag.retrieval.retrieval_filters import FilterBuilder, RetrievalFilter
from knowledge_engine.rag.retrieval.reranker import NoOpReranker, Reranker, RerankResult
from knowledge_engine.rag.vectorstore.chroma_repository import ChromaRepository, QueryResult
 
logger = get_logger(__name__)
 
 
# ─── Exceptions ──────────────────────────────────────────────────────────────
 
 
class RetrievalError(APIError):
    """Raised when the retrieval pipeline fails."""
 
 
# ─── Result Dataclass ────────────────────────────────────────────────────────
 
 
@dataclass
class RetrievalResult:
    """
    A single retrieved chunk, ready for the generation layer.
 
    Combines fields from QueryResult (similarity search) and RerankResult
    (optional reranking) into one flat, convenient structure.
 
    Fields:
        chunk_id:      Stable identifier for the chunk.
        text:          Original chunk text (not the normalised embedding text).
        score:         Final ranking score (rerank_score if reranked, else
                       cosine similarity score from the vector store).
        similarity_score: Raw cosine similarity from the vector store (before reranking).
        document_id:   Source document identifier.
        source_file:   Path to the original source file.
        chunk_index:   Position of this chunk within its document.
        page_number:   Estimated page number.
        section:       Section heading (empty string if not detected).
        char_start:    Character offset start in cleaned text.
        char_end:      Character offset end in cleaned text.
        word_count:    Number of words.
        quality_score: Document quality score (0.0–1.0).
        metadata:      Full raw metadata dict from ChromaDB (for debugging).
    """
 
    chunk_id: str
    text: str
    score: float
    similarity_score: float
    document_id: str
    source_file: str
    chunk_index: int
    page_number: int
    section: str
    char_start: int
    char_end: int
    word_count: int
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
 
    def __repr__(self) -> str:
        return (
            f"RetrievalResult(score={self.score:.4f}, "
            f"doc={self.document_id!r}, "
            f"page={self.page_number}, "
            f"chunk={self.chunk_index}, "
            f"text={self.text[:60]!r}{'...' if len(self.text) > 60 else ''})"
        )
 
 
# ─── Retriever Config ────────────────────────────────────────────────────────
 
 
@dataclass
class RetrieverConfig:
    """
    Runtime configuration for the Retriever.
 
    Fields:
        top_k:                   Default number of results to return.
        score_threshold:         Default minimum similarity score (0.0–1.0).
        max_chunks_per_document: Cap on how many chunks from the same document
                                 can appear in one result set (0 = unlimited).
        rerank_top_k:            How many results to return after reranking.
                                 None → same as top_k.
    """
 
    top_k: int = 5
    score_threshold: float = 0.0
    max_chunks_per_document: int = 0
    rerank_top_k: int | None = None
 
 
# ─── Retriever ───────────────────────────────────────────────────────────────
 
 
class Retriever:
    """
    RAG retrieval pipeline: query → embed → search → filter → (rerank) → results.
 
    Dependency-injected: accepts any EmbeddingProvider, ChromaRepository,
    and Reranker so tests can replace any layer without hitting real APIs.
 
    Usage:
        # Production (uses default OpenAI + ChromaDB + NoOp reranker)
        retriever = get_retriever()
        results   = retriever.search("What exclusions apply to flood damage?")
 
        # With filters
        filt    = RetrievalFilter(document_id="policy_v3", min_page=2, max_page=10)
        results = retriever.search("flood exclusion", filter_=filt, top_k=3)
 
        # With per-document cap
        results = retriever.search("deductible", max_chunks_per_document=2)
    """
 
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        repository: ChromaRepository,
        reranker: Reranker | None = None,
        config: RetrieverConfig | None = None,
    ) -> None:
        """
        Initialise the retriever.
 
        Args:
            embedding_provider: Provider for query embedding.
            repository:         ChromaDB repository to search.
            reranker:           Optional reranker (defaults to NoOpReranker).
            config:             Optional config override (defaults from Settings).
        """
        if config is None:
            settings = get_settings()
            config = RetrieverConfig(
                top_k=settings.TOP_K,
                score_threshold=settings.SIMILARITY_THRESHOLD,
            )
 
        self._provider = embedding_provider
        self._repo = repository
        self._reranker: Reranker = reranker or NoOpReranker()
        self._config = config
 
        logger.info(
            "Retriever initialised: model=%s, top_k=%d, threshold=%.2f, reranker=%s",
            self._provider.model_name,
            self._config.top_k,
            self._config.score_threshold,
            self._reranker.model_name,
        )
 
    # ── Public API ───────────────────────────────────────────────────────
 
    def search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter_: RetrievalFilter | None = None,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant chunks for a natural-language query.
 
        Args:
            query:                   User query string.
            top_k:                   Override the default number of results.
            score_threshold:         Override the default minimum similarity score.
            filter_:                 Metadata filter (document, page, section, …).
            max_chunks_per_document: Override per-document chunk cap.
 
        Returns:
            RetrievalResult list, sorted by final score descending.
 
        Raises:
            RetrievalError: If embedding fails or the vector store is unavailable.
        """
        if not query or not query.strip():
            logger.warning("search called with empty query — returning []")
            return []
 
        effective_top_k = top_k if top_k is not None else self._config.top_k
        effective_threshold = (
            score_threshold if score_threshold is not None else self._config.score_threshold
        )
        effective_doc_cap = (
            max_chunks_per_document
            if max_chunks_per_document is not None
            else self._config.max_chunks_per_document
        )
 
        logger.debug(
            "search: query=%r, top_k=%d, threshold=%.2f",
            query[:80],
            effective_top_k,
            effective_threshold,
        )
 
        # Step 1: Embed the query
        query_embedding = self._embed_query(query)
 
        # Step 2: Build where clause from filter
        where = FilterBuilder.build(filter_)
 
        # Step 3: Vector similarity search
        # Request more than top_k so per-document cap has headroom
        fetch_k = effective_top_k * 3 if effective_doc_cap > 0 else effective_top_k
        raw_results = self._repo.query(
            embedding=query_embedding,
            top_k=max(fetch_k, effective_top_k),
            score_threshold=effective_threshold,
            where=where,
        )
 
        # Step 4: Apply per-document cap
        if effective_doc_cap > 0:
            raw_results = _cap_per_document(raw_results, effective_doc_cap)
 
        # Step 5: Trim to top_k
        raw_results = raw_results[:effective_top_k]
 
        # Step 6: Rerank
        reranked = self._reranker.rerank(query, raw_results, top_k=effective_top_k)
 
        # Step 7: Map to RetrievalResult
        results = [_to_retrieval_result(r) for r in reranked]
 
        logger.info(
            "search complete: query=%r, returned=%d/%d (threshold=%.2f)",
            query[:60],
            len(results),
            len(raw_results),
            effective_threshold,
        )
        return results
 
    def search_with_filter_dict(
        self,
        query: str,
        where: dict[str, Any] | None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Lower-level search accepting a raw ChromaDB where dict.
 
        Use this when FilterBuilder.build() does not cover your use case.
        Prefer search() + RetrievalFilter for normal use.
 
        Args:
            query:           User query.
            where:           Raw ChromaDB where clause dict (or None).
            top_k:           Override top_k.
            score_threshold: Override score threshold.
 
        Returns:
            RetrievalResult list.
        """
        effective_top_k = top_k if top_k is not None else self._config.top_k
        effective_threshold = (
            score_threshold if score_threshold is not None else self._config.score_threshold
        )
 
        query_embedding = self._embed_query(query)
        raw_results = self._repo.query(
            embedding=query_embedding,
            top_k=effective_top_k,
            score_threshold=effective_threshold,
            where=where,
        )
        reranked = self._reranker.rerank(query, raw_results, top_k=effective_top_k)
        return [_to_retrieval_result(r) for r in reranked]
 
    # ── Internal helpers ─────────────────────────────────────────────────
 
    def _embed_query(self, query: str) -> list[float]:
        """Embed the query string and return the vector."""
        try:
            result = self._provider.embed_text(query)
            logger.debug("Query embedded: tokens=%d, dims=%d", result.token_count, result.dimensions)
            return result.embedding
        except EmbeddingError as exc:
            raise RetrievalError(f"Failed to embed query: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(f"Unexpected error embedding query: {exc}") from exc
 
 
# ─── Helper Functions ─────────────────────────────────────────────────────────
 
 
def _cap_per_document(
    results: list[QueryResult],
    max_per_doc: int,
) -> list[QueryResult]:
    """
    Limit the number of results from any single document.
 
    Preserves relative score ordering within and across documents.
    Chunks without a document_id in metadata are treated as a single bucket.
 
    Args:
        results:     Already score-sorted QueryResult list.
        max_per_doc: Maximum chunks allowed from any one document.
 
    Returns:
        Filtered list (same score ordering, trimmed per document).
    """
    counts: dict[str, int] = {}
    filtered: list[QueryResult] = []
 
    for r in results:
        doc_id = r.metadata.get("document_id", "__unknown__")
        if counts.get(doc_id, 0) < max_per_doc:
            filtered.append(r)
            counts[doc_id] = counts.get(doc_id, 0) + 1
 
    logger.debug(
        "_cap_per_document: %d → %d results (max_per_doc=%d)",
        len(results),
        len(filtered),
        max_per_doc,
    )
    return filtered
 
 
def _to_retrieval_result(r: RerankResult) -> RetrievalResult:
    """Map a RerankResult to the flat RetrievalResult exposed to callers."""
    meta = r.metadata
    return RetrievalResult(
        chunk_id=r.chunk_id,
        text=r.text,
        score=r.rerank_score,
        similarity_score=r.original.score,
        document_id=str(meta.get("document_id", "")),
        source_file=str(meta.get("source_file", "")),
        chunk_index=int(meta.get("chunk_index", -1)),
        page_number=int(meta.get("page_number", -1)),
        section=str(meta.get("section", "")),
        char_start=int(meta.get("char_start", 0)),
        char_end=int(meta.get("char_end", 0)),
        word_count=int(meta.get("word_count", 0)),
        quality_score=float(meta.get("quality_score", 0.0)),
        metadata=meta,
    )
 
 
# ─── Factory ─────────────────────────────────────────────────────────────────
 
 
@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """
    Return a cached Retriever configured from Settings.
 
    Uses the same factory pattern as get_settings() / get_token_calculator().
    Imports are deferred to avoid circular imports at module load time.
 
    Returns:
        Retriever singleton.
    """
    from knowledge_engine.adapters.openai.openai_embeddings import (
        get_openai_embedding_provider,
    )
    from knowledge_engine.rag.retrieval.reranker import get_default_reranker
    from knowledge_engine.rag.vectorstore.chroma_repository import get_chroma_repository
 
    return Retriever(
        embedding_provider=get_openai_embedding_provider(),
        repository=get_chroma_repository(),
        reranker=get_default_reranker(),
    )
