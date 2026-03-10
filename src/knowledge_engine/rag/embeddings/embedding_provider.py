"""
Embedding provider abstraction for the GuardianRAG system (Enterprise Edition)
 
Design Pattern: Provider / Strategy
  - EmbeddingProvider defines the contract all embedding backends must implement
  - Concrete implementations (OpenAI, local models, etc.) are fully swappable
  - EmbeddingResult carries the full vector + provenance metadata
 
Responsibilities:
  - Define the embedding interface (embed_text, embed_batch)
  - Define structured result dataclasses
  - Define shared exceptions
 
Usage:
    # Use via a concrete implementation (e.g. OpenAIEmbeddingProvider)
    provider: EmbeddingProvider = OpenAIEmbeddingProvider()
    result = provider.embed_text("Insurance policy section 3.1")
    vector = result.embedding  # List[float], ready for ChromaDB
"""
from __future__ import annotations
 
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
 
from knowledge_engine.core.exceptions import APIError
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
# ─── Exceptions ─────────────────────────────────────────────────────────────
 
 
class EmbeddingError(APIError):
    """Raised when text embedding fails."""
 
 
class BatchEmbeddingError(EmbeddingError):
    """Raised when batch embedding fails, with partial-result support."""
 
    def __init__(self, message: str, partial_results: list["EmbeddingResult"] | None = None) -> None:
        super().__init__(message)
        self.partial_results: list[EmbeddingResult] = partial_results or []
 
 
# ─── Result Dataclasses ──────────────────────────────────────────────────────
 
 
@dataclass
class EmbeddingResult:
    """
    Structured result from a single text embedding call.
 
    Fields:
        text:          The original input text (preserved for display / debug).
        embedding:     The embedding vector as a list of floats.
        model:         Model name used to produce this embedding.
        token_count:   Approximate number of tokens consumed (from API response
                       when available, else -1 if unknown).
        dimensions:    Length of the embedding vector.
        metadata:      Arbitrary key/value pairs from the provider (e.g.
                       finish_reason, usage breakdown).
    """
 
    text: str
    embedding: list[float]
    model: str
    token_count: int = -1
    dimensions: int = field(init=False)
    metadata: dict[str, Any] = field(default_factory=dict)
 
    def __post_init__(self) -> None:
        self.dimensions = len(self.embedding)
 
    def __repr__(self) -> str:
        return (
            f"EmbeddingResult(model={self.model!r}, "
            f"dimensions={self.dimensions}, "
            f"tokens={self.token_count}, "
            f"text={self.text[:40]!r}{'...' if len(self.text) > 40 else ''})"
        )
 
 
@dataclass
class BatchEmbeddingResult:
    """
    Structured result from a batch embedding call.
 
    Fields:
        results:        One EmbeddingResult per input text, in input order.
        model:          Model name used for the entire batch.
        total_tokens:   Sum of token_count across all results (-1 if unknown).
        metadata:       Arbitrary provider-level metadata for the batch call.
    """
 
    results: list[EmbeddingResult]
    model: str
    total_tokens: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)
 
    def __len__(self) -> int:
        return len(self.results)
 
    def __iter__(self):
        return iter(self.results)
 
    def __repr__(self) -> str:
        return (
            f"BatchEmbeddingResult(model={self.model!r}, "
            f"count={len(self.results)}, "
            f"total_tokens={self.total_tokens})"
        )
 
    @property
    def embeddings(self) -> list[list[float]]:
        """Convenience: return all embedding vectors in input order."""
        return [r.embedding for r in self.results]
 
 
# ─── Text Normalisation Helper ───────────────────────────────────────────────
 
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
 
 
def normalize_for_embedding(text: str) -> str:
    """
    Light-touch text normalisation applied BEFORE embedding.
 
    Purpose:
        Improve embedding quality by reducing noise (collapsed whitespace,
        trimmed excess blank lines) while keeping the original chunk text
        intact for display to end users.
 
    Rules applied:
        1. Collapse multiple spaces/tabs → single space (per line).
        2. Collapse 3+ consecutive newlines → double newline (paragraph break).
        3. Strip leading/trailing whitespace.
 
    This is intentionally minimal.  Heavier normalisation (unicode repair,
    soft-hyphen removal, header/footer stripping) is handled in Week 2 and
    must NOT be repeated here to avoid double-processing.
 
    Args:
        text: Chunk text as stored in the JSONL file.
 
    Returns:
        Lightly normalised text suitable for the embedding model.
    """
    if not text:
        return text
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
 
 
# ─── Abstract Base Class ─────────────────────────────────────────────────────
 
 
class EmbeddingProvider(ABC):
    """
    Abstract base class for all embedding providers.
 
    Concrete implementations must override:
        embed_text(text)    → EmbeddingResult
        embed_batch(texts)  → BatchEmbeddingResult
        model_name          → str  (property)
        dimensions          → int  (property)
 
    Design notes:
        - Callers should depend on EmbeddingProvider, not on a concrete class.
        - The provider is responsible for calling normalize_for_embedding()
          before sending text to the model, while preserving the original text
          in EmbeddingResult.text.
        - Implementations must be thread-safe; the repository layer may call
          them from multiple threads during bulk indexing.
    """
 
    # ── Abstract interface ────────────────────────────────────────────────
 
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model (e.g. 'text-embedding-3-small')."""
        ...
 
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        ...
 
    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Embed a single piece of text.
 
        Args:
            text: The raw chunk text to embed.  The provider will normalise
                  it internally before sending to the model.
 
        Returns:
            EmbeddingResult with the embedding vector and provenance metadata.
 
        Raises:
            EmbeddingError: If the embedding call fails for any reason.
        """
        ...
 
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """
        Embed a list of texts in a single API call where possible.
 
        Args:
            texts: List of raw chunk texts.  Order is preserved in results.
 
        Returns:
            BatchEmbeddingResult where results[i] corresponds to texts[i].
 
        Raises:
            BatchEmbeddingError: If the batch call fails.  partial_results
                may contain embeddings that succeeded before the failure.
        """
        ...
 
    # ── Concrete helpers (available to all subclasses) ────────────────────
 
    def _normalize(self, text: str) -> str:
        """Apply pre-embedding normalisation (delegates to module function)."""
        return normalize_for_embedding(text)
 
    def embed_texts_chunked(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> BatchEmbeddingResult:
        """
        Embed a large list of texts by splitting into smaller batches.
 
        Useful when the provider limits batch size (OpenAI: 2048 per call).
        Each sub-batch is embedded with embed_batch(); results are merged.
 
        Args:
            texts:      All texts to embed.
            batch_size: Maximum texts per sub-batch (default 100).
 
        Returns:
            Single BatchEmbeddingResult covering all texts.
 
        Raises:
            BatchEmbeddingError: On the first sub-batch that fails.
        """
        if not texts:
            return BatchEmbeddingResult(results=[], model=self.model_name, total_tokens=0)
 
        all_results: list[EmbeddingResult] = []
        total_tokens = 0
 
        for i in range(0, len(texts), batch_size):
            sub_batch = texts[i : i + batch_size]
            logger.debug(
                "Embedding sub-batch %d-%d of %d texts",
                i,
                i + len(sub_batch) - 1,
                len(texts),
            )
            batch_result = self.embed_batch(sub_batch)
            all_results.extend(batch_result.results)
            if batch_result.total_tokens >= 0:
                total_tokens += batch_result.total_tokens
 
        logger.info(
            "Embedded %d texts in %d batches (model=%s, total_tokens=%d)",
            len(texts),
            (len(texts) + batch_size - 1) // batch_size,
            self.model_name,
            total_tokens,
        )
 
        return BatchEmbeddingResult(
            results=all_results,
            model=self.model_name,
            total_tokens=total_tokens,
        )
