"""
Reranker interface for the GuardianRAG system (Enterprise Edition)
 
Defines the contract for cross-encoder reranking of retrieved chunks.
 
Status: INTERFACE ONLY — no model is implemented here.
 
Purpose:
  After the initial vector similarity search (retriever.py), an optional
  reranking step can improve result quality using a cross-encoder model
  that jointly scores the (query, chunk) pair — more expensive but more
  accurate than bi-encoder similarity alone.
 
Design:
  - Reranker is the abstract base class all rerankers must implement.
  - NoOpReranker passes results through unchanged (default production value
    until a real cross-encoder is wired in).
  - RerankResult carries the original QueryResult plus the reranker's score.
 
Implementation note for future engineers:
  To add a concrete cross-encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2
  via HuggingFace sentence-transformers, or Cohere Rerank API):
 
  1. Subclass Reranker.
  2. Override rerank(query, results, top_k) → list[RerankResult].
  3. Wire it into Retriever.__init__ via dependency injection.
  4. No other module needs to change.
 
Usage (current):
    reranker = NoOpReranker()
    reranked = reranker.rerank(query="coverage for flood damage", results=raw_results)
 
Usage (future):
    reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranked = reranker.rerank(query=..., results=raw_results, top_k=5)
"""
from __future__ import annotations
 
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
 
from knowledge_engine.core.logging_config import get_logger
from knowledge_engine.rag.vectorstore.chroma_repository import QueryResult
 
logger = get_logger(__name__)
 
 
# ─── Result Dataclass ────────────────────────────────────────────────────────
 
 
@dataclass
class RerankResult:
    """
    A single result after reranking.
 
    Fields:
        original:         The QueryResult from the vector store.
        rerank_score:     Score from the reranker model (higher = more relevant).
                          For NoOpReranker this equals original.score.
        rerank_model:     Name of the reranker model used (or "noop").
        original_rank:    Zero-based rank BEFORE reranking.
        rerank_rank:      Zero-based rank AFTER reranking.
    """
 
    original: QueryResult
    rerank_score: float
    rerank_model: str
    original_rank: int = 0
    rerank_rank: int = 0
 
    # ── Convenience pass-through properties ──────────────────────────────
 
    @property
    def text(self) -> str:
        """Chunk text (from original QueryResult)."""
        return self.original.text
 
    @property
    def chunk_id(self) -> str:
        return self.original.chunk_id
 
    @property
    def metadata(self) -> dict[str, Any]:
        return self.original.metadata
 
    def get(self, key: str, default: Any = None) -> Any:
        return self.original.get(key, default)
 
    def __repr__(self) -> str:
        return (
            f"RerankResult(rerank_rank={self.rerank_rank}, "
            f"rerank_score={self.rerank_score:.4f}, "
            f"original_score={self.original.score:.4f}, "
            f"text={self.text[:50]!r})"
        )
 
 
# ─── Abstract Base Class ─────────────────────────────────────────────────────
 
 
class Reranker(ABC):
    """
    Abstract interface for result rerankers.
 
    All implementations must override rerank().
    """
 
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the reranker model."""
        ...
 
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[QueryResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """
        Rerank a list of QueryResults for a given query.
 
        Args:
            query:   The original user query string.
            results: Candidate results from the vector similarity search,
                     in their original score order.
            top_k:   Return at most this many results.  None → return all.
 
        Returns:
            RerankResult list sorted by rerank_score descending.
            Each element carries both the original similarity score and the
            reranker score so callers can compare the two rankings.
        """
        ...
 
 
# ─── NoOp Implementation ─────────────────────────────────────────────────────
 
 
class NoOpReranker(Reranker):
    """
    Pass-through reranker that preserves the vector similarity ordering.
 
    Used as the default until a real cross-encoder is integrated.
 
    The rerank_score is set equal to the original cosine similarity score
    so downstream callers can treat all rerankers uniformly.
    """
 
    _MODEL_NAME = "noop"
 
    @property
    def model_name(self) -> str:
        return self._MODEL_NAME
 
    def rerank(
        self,
        query: str,
        results: list[QueryResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """
        Return results unchanged, wrapped in RerankResult.
 
        Args:
            query:   Unused in the no-op implementation.
            results: QueryResults from the vector store.
            top_k:   Truncate to this many results.
 
        Returns:
            RerankResults in the same order as input (already sorted by
            similarity score from the vector store).
        """
        reranked = [
            RerankResult(
                original=r,
                rerank_score=r.score,
                rerank_model=self._MODEL_NAME,
                original_rank=i,
                rerank_rank=i,
            )
            for i, r in enumerate(results)
        ]
 
        if top_k is not None:
            reranked = reranked[:top_k]
 
        logger.debug(
            "NoOpReranker: %d results passed through (top_k=%s)",
            len(reranked),
            top_k,
        )
        return reranked
 
 
# ─── Factory ─────────────────────────────────────────────────────────────────
 
 
def get_default_reranker() -> Reranker:
    """
    Return the default reranker instance.
 
    Currently returns NoOpReranker.
    Replace this function body (or override via DI) to activate a real model.
    """
    return NoOpReranker()
