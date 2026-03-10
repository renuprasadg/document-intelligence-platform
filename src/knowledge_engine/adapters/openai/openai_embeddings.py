"""
OpenAI embedding adapter for the GuardianRAG system (Enterprise Edition)
 
Implements EmbeddingProvider using the OpenAI Embeddings API.
 
Responsibilities:
  - Read API key from Settings.OPENAI_API_KEY (via get_settings() factory)
  - Send text to the OpenAI embeddings endpoint
  - Map API response → EmbeddingResult / BatchEmbeddingResult
  - Log token usage for cost tracking (integrates with Week 1 token_cost module)
  - Retry on transient API errors (rate limits, 5xx)
 
Supported models:
  - text-embedding-3-small  (default, cheapest, 1536-dim)
  - text-embedding-3-large  (3072-dim, highest quality)
  - text-embedding-ada-002  (legacy, 1536-dim)
 
Usage:
    provider = get_openai_embedding_provider()        # cached singleton
    result   = provider.embed_text("Policy section…")
    batch    = provider.embed_batch(["text1", "text2"])
"""
from __future__ import annotations
 
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
 
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
 
from knowledge_engine.core.config import get_settings
from knowledge_engine.core.logging_config import get_logger
from knowledge_engine.rag.embeddings.embedding_provider import (
    BatchEmbeddingError,
    BatchEmbeddingResult,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingResult,
)
 
logger = get_logger(__name__)
 
# ─── Known model dimensions ──────────────────────────────────────────────────
 
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
 
_DEFAULT_DIMENSIONS = 1536
 
 
# ─── Provider config dataclass ───────────────────────────────────────────────
 
 
@dataclass
class OpenAIEmbeddingConfig:
    """
    Runtime configuration for the OpenAI embedding provider.
 
    Fields:
        model:          OpenAI embedding model name.
        max_retries:    Max automatic retries on rate-limit / transient errors.
        retry_delay_s:  Base delay in seconds between retries (doubles each attempt).
        timeout_s:      Per-request timeout in seconds.
        batch_size:     Maximum texts per API call (OpenAI hard limit: 2048).
        extra_headers:  Optional extra HTTP headers forwarded to the OpenAI client.
    """
 
    model: str = "text-embedding-3-small"
    max_retries: int = 3
    retry_delay_s: float = 1.0
    timeout_s: float = 30.0
    batch_size: int = 100
    extra_headers: dict[str, str] = field(default_factory=dict)
 
 
# ─── Concrete provider ───────────────────────────────────────────────────────
 
 
class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI-backed embedding provider.
 
    Enterprise features:
        - Settings factory integration (reads OPENAI_API_KEY, EMBEDDING_MODEL)
        - Configurable model, batch size, retry logic
        - Token usage logged for cost tracking (see Week 1 token_cost.py)
        - Thread-safe (OpenAI client is thread-safe per SDK docs)
 
    Usage:
        provider = OpenAIEmbeddingProvider()
        result   = provider.embed_text("What is covered under section 4?")
        print(result.embedding[:5])   # first 5 dimensions
        print(result.token_count)     # tokens consumed
    """
 
    def __init__(self, config: OpenAIEmbeddingConfig | None = None) -> None:
        """
        Initialise the provider.
 
        Args:
            config: Optional config override.  If None, defaults are read from
                    the Settings factory (OPENAI_API_KEY, EMBEDDING_MODEL).
        """
        settings = get_settings()
 
        if config is None:
            config = OpenAIEmbeddingConfig(model=settings.EMBEDDING_MODEL)
 
        self._config = config
        self._client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=config.timeout_s,
            default_headers=config.extra_headers or {},
        )
        self._dimensions = _MODEL_DIMENSIONS.get(config.model, _DEFAULT_DIMENSIONS)
 
        logger.info(
            "OpenAIEmbeddingProvider initialised: model=%s, dimensions=%d, batch_size=%d",
            config.model,
            self._dimensions,
            config.batch_size,
        )
 
    # ── EmbeddingProvider abstract interface ─────────────────────────────
 
    @property
    def model_name(self) -> str:
        return self._config.model
 
    @property
    def dimensions(self) -> int:
        return self._dimensions
 
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Embed a single text string.
 
        Internally calls embed_batch([text]) to reuse retry and logging logic.
 
        Args:
            text: Raw chunk text (will be normalised before embedding).
 
        Returns:
            EmbeddingResult with vector + provenance.
 
        Raises:
            EmbeddingError: On API failure after all retries exhausted.
        """
        try:
            batch = self.embed_batch([text])
            return batch.results[0]
        except BatchEmbeddingError as exc:
            raise EmbeddingError(f"embed_text failed: {exc}") from exc
 
    def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """
        Embed a list of texts in a single OpenAI API call.
 
        Args:
            texts: List of raw chunk texts (will each be normalised).
 
        Returns:
            BatchEmbeddingResult in input order.
 
        Raises:
            BatchEmbeddingError: After max_retries exhausted.
        """
        if not texts:
            return BatchEmbeddingResult(results=[], model=self.model_name, total_tokens=0)
 
        # Normalise before sending to the model (keep originals for display)
        normalized = [self._normalize(t) for t in texts]
 
        for attempt in range(1, self._config.max_retries + 1):
            try:
                return self._call_api(texts, normalized, attempt)
            except RateLimitError as exc:
                wait = self._config.retry_delay_s * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI rate limit hit (attempt %d/%d). Waiting %.1fs: %s",
                    attempt,
                    self._config.max_retries,
                    wait,
                    exc,
                )
                if attempt == self._config.max_retries:
                    raise BatchEmbeddingError(
                        f"Rate limit after {attempt} attempts: {exc}"
                    ) from exc
                time.sleep(wait)
            except APIConnectionError as exc:
                wait = self._config.retry_delay_s * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI connection error (attempt %d/%d). Waiting %.1fs: %s",
                    attempt,
                    self._config.max_retries,
                    wait,
                    exc,
                )
                if attempt == self._config.max_retries:
                    raise BatchEmbeddingError(
                        f"Connection error after {attempt} attempts: {exc}"
                    ) from exc
                time.sleep(wait)
            except APIStatusError as exc:
                # 5xx → retry; 4xx (except 429) → fail immediately
                if exc.status_code >= 500:
                    wait = self._config.retry_delay_s * (2 ** (attempt - 1))
                    logger.warning(
                        "OpenAI server error %d (attempt %d/%d). Waiting %.1fs.",
                        exc.status_code,
                        attempt,
                        self._config.max_retries,
                        wait,
                    )
                    if attempt == self._config.max_retries:
                        raise BatchEmbeddingError(
                            f"Server error {exc.status_code} after {attempt} attempts: {exc}"
                        ) from exc
                    time.sleep(wait)
                else:
                    raise BatchEmbeddingError(
                        f"OpenAI API error {exc.status_code}: {exc}"
                    ) from exc
 
        # Should be unreachable, but satisfies type checkers
        raise BatchEmbeddingError("embed_batch failed: exhausted retries")
 
    # ── Internal helpers ─────────────────────────────────────────────────
 
    def _call_api(
        self,
        original_texts: list[str],
        normalized_texts: list[str],
        attempt: int,
    ) -> BatchEmbeddingResult:
        """Issue the actual API call and map response to BatchEmbeddingResult."""
        logger.debug(
            "Calling OpenAI embeddings API: model=%s, texts=%d, attempt=%d",
            self._config.model,
            len(normalized_texts),
            attempt,
        )
 
        response = self._client.embeddings.create(
            input=normalized_texts,
            model=self._config.model,
        )
 
        total_tokens: int = response.usage.total_tokens if response.usage else -1
 
        # OpenAI returns embeddings sorted by `index` — sort defensively
        sorted_data = sorted(response.data, key=lambda d: d.index)
 
        results: list[EmbeddingResult] = []
        for orig_text, data_item in zip(original_texts, sorted_data, strict=True):
            results.append(
                EmbeddingResult(
                    text=orig_text,
                    embedding=data_item.embedding,
                    model=self._config.model,
                    token_count=total_tokens // len(original_texts) if total_tokens > 0 else -1,
                    metadata={"index": data_item.index, "object": data_item.object},
                )
            )
 
        logger.info(
            "Embedded %d texts: model=%s, total_tokens=%d",
            len(results),
            self._config.model,
            total_tokens,
        )
 
        return BatchEmbeddingResult(
            results=results,
            model=self._config.model,
            total_tokens=total_tokens,
            metadata=_usage_to_dict(response.usage),
        )
 
 
# ─── Factory ─────────────────────────────────────────────────────────────────
 
 
@lru_cache(maxsize=1)
def get_openai_embedding_provider() -> OpenAIEmbeddingProvider:
    """
    Return a cached OpenAIEmbeddingProvider configured from Settings.
 
    Uses the same factory / lru_cache pattern as get_settings() so there is
    only one client instance per process.
 
    Returns:
        OpenAIEmbeddingProvider singleton.
    """
    return OpenAIEmbeddingProvider()
 
 
# ─── Helpers ─────────────────────────────────────────────────────────────────
 
 
def _usage_to_dict(usage: Any) -> dict[str, Any]:
    """Convert OpenAI Usage object to a plain dict for metadata storage."""
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
