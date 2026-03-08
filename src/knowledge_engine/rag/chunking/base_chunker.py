"""
Base chunker abstract class + sliding window chunker (Enterprise Edition)
 
Design Pattern: Strategy Pattern
  - BaseChunker defines the contract
  - Concrete implementations are swappable
  - ChunkResult carries full metadata
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator, List, Optional
 
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
@dataclass
class Chunk:
    """A single text chunk with metadata"""
    text: str
    chunk_index: int              # 0-based index in document
    char_start: int               # Start character offset in original text
    char_end: int                 # End character offset
    word_count: int = 0
    source: str = ""              # Source document path
    extra: dict = field(default_factory=dict)  # Chunker-specific metadata
 
    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.text.split())
 
    def __len__(self) -> int:
        return len(self.text)
 
 
@dataclass
class ChunkResult:
    """Result of a chunking operation"""
    source: str
    chunks: List[Chunk] = field(default_factory=list)
    total_chunks: int = 0
    total_words: int = 0
    chunker_name: str = ""
    config_summary: str = ""
 
    def __post_init__(self):
        if self.total_chunks == 0:
            self.total_chunks = len(self.chunks)
        if self.total_words == 0:
            self.total_words = sum(c.word_count for c in self.chunks)
 
    def __iter__(self) -> Iterator[Chunk]:
        return iter(self.chunks)
 
    def __len__(self) -> int:
        return len(self.chunks)
 
    def texts(self) -> List[str]:
        """Return list of chunk texts (for embedding)."""
        return [c.text for c in self.chunks]
 
 
@dataclass
class ChunkConfig:
    """Base configuration for all chunkers"""
    chunk_size: int = 512      # Target chunk size in words
    chunk_overlap: int = 50    # Overlap in words between chunks
    min_chunk_size: int = 20   # Discard chunks smaller than this
    max_chunk_size: int = 1024 # Hard maximum in words
 
 
class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
 
    All chunkers must implement chunk() which returns a ChunkResult.
    """
 
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
 
    @abstractmethod
    def chunk(self, text: str, source: str = "") -> ChunkResult:
        """
        Split text into chunks.
 
        Args:
            text: Cleaned text to chunk
            source: Source identifier (e.g. filename)
 
        Returns:
            ChunkResult containing list of Chunk objects
        """
        ...
 
    @property
    @abstractmethod
    def name(self) -> str:
        """Chunker name for logging and metadata."""
        ...
 
    def _make_chunk(self, text: str, index: int, start: int, source: str = "") -> Chunk:
        """Helper: construct a Chunk, stripping whitespace."""
        clean = text.strip()
        end = start + len(text)
        return Chunk(text=clean, chunk_index=index, char_start=start, char_end=end, source=source)
 
    def _is_valid(self, chunk: Chunk) -> bool:
        """Check chunk meets minimum size requirements."""
        return chunk.word_count >= self.config.min_chunk_size
 
 
class SlidingWindowChunker(BaseChunker):
    """
    Fixed-size sliding window chunker.
 
    Splits text into chunks of `chunk_size` words with `chunk_overlap` word overlap.
    Simple and reliable - use as a baseline or when semantic chunking is overkill.
 
    Usage:
        chunker = get_sliding_window_chunker()
        result = chunker.chunk(clean_text, source="doc.pdf")
    """
 
    @property
    def name(self) -> str:
        return "SlidingWindowChunker"
 
    def chunk(self, text: str, source: str = "") -> ChunkResult:
        """
        Split text into overlapping word-count windows.
 
        Args:
            text: Cleaned document text
            source: Source document identifier
 
        Returns:
            ChunkResult with sliding-window chunks.
 
        Note on char_start / char_end:
            Offsets are computed by searching for the first word of each chunk
            in the original text starting from the previous chunk end.
            They are accurate for non-overlapping steps; overlapping windows
            share characters, so treat offsets as the start of the new content.
        """
        if not text or not text.strip():
            return ChunkResult(source=source, chunker_name=self.name)
 
        words = text.split()
        step = max(1, self.config.chunk_size - self.config.chunk_overlap)
        chunks: List[Chunk] = []
        search_from = 0   # cursor into original text for accurate char_start
 
        for i in range(0, len(words), step):
            window_words = words[i : i + self.config.chunk_size]
            chunk_text = " ".join(window_words)
 
            # Find accurate char_start by locating first word in original text
            first_word = window_words[0] if window_words else ""
            char_start = text.find(first_word, search_from)
            if char_start == -1:
                char_start = search_from  # fallback
            char_end = char_start + len(chunk_text)
 
            chunk = Chunk(
                text=chunk_text.strip(),
                chunk_index=len(chunks),
                char_start=char_start,
                char_end=char_end,
                source=source,
            )
            if self._is_valid(chunk):
                chunks.append(chunk)
 
            # Advance search cursor by the non-overlapping step
            step_text = " ".join(words[i : i + step])
            step_pos = text.find(step_text, search_from)
            if step_pos != -1:
                search_from = step_pos + len(step_text)
 
            if i + self.config.chunk_size >= len(words):
                break
 
        logger.debug(
            "%s produced %d chunks from %d words",
            self.name, len(chunks), len(words)
        )
 
        return ChunkResult(
            source=source,
            chunks=chunks,
            chunker_name=self.name,
            config_summary=f"size={self.config.chunk_size},overlap={self.config.chunk_overlap}",
        )
 
 
@lru_cache(maxsize=1)
def get_sliding_window_chunker() -> SlidingWindowChunker:
    """Factory function - returns cached SlidingWindowChunker."""
    return SlidingWindowChunker()
