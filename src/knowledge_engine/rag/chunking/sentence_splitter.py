"""
Sentence-boundary-aware chunker (Enterprise Edition)
 
Strategy:
  1. Split text into sentences using regex
  2. Accumulate sentences into a chunk until chunk_size words is reached
  3. Start new chunk (with optional sentence overlap)
 
No external NLP dependencies - pure regex.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional
 
from knowledge_engine.rag.chunking.base_chunker import (
    BaseChunker, Chunk, ChunkConfig, ChunkResult,
)
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
@dataclass
class SentenceChunkConfig(ChunkConfig):
    """Extended config for SentenceSplitter"""
    sentence_overlap: int = 1   # Number of sentences to carry over to next chunk
    min_sentence_words: int = 3 # Ignore very short fragments
 
 
# Sentence boundary: end of sentence punctuation followed by space/newline
# Does not split on abbreviations like "e.g.", "U.S.A.", "Dr.", etc.
_SENT_END = re.compile(
    r"(?<![A-Z][a-z])"      # Not after capital + lowercase (abbreviation)
    r"(?<!\d)"              # Not after a digit (e.g. "2.5 GHz")
    r"[.!?]"                 # Sentence-ending punctuation
    r"[\"'\)\]]?"        # Optional closing quote/bracket
    r"(?=\s+[A-Z0-9])",     # Followed by whitespace + capital/digit
    re.VERBOSE
)
 
 
def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.
 
    Args:
        text: Cleaned text to split
 
    Returns:
        List of sentence strings
    """
    # Insert a split marker at sentence boundaries
    marked = _SENT_END.sub(lambda m: m.group() + "\x00", text)
    sentences = [s.strip() for s in marked.split("\x00") if s.strip()]
    return sentences
 
 
class SentenceSplitter(BaseChunker):
    """
    Accumulates sentences into chunks without breaking mid-sentence.
 
    Usage:
        splitter = get_sentence_splitter()
        result = splitter.chunk(clean_text, source="report.pdf")
    """
 
    def __init__(self, config: Optional[SentenceChunkConfig] = None):
        super().__init__(config or SentenceChunkConfig())
        self._config: SentenceChunkConfig = self.config  # type: ignore
 
    @property
    def name(self) -> str:
        return "SentenceSplitter"
 
    def chunk(self, text: str, source: str = "") -> ChunkResult:
        """
        Chunk text by accumulating sentences to target word count.
 
        Args:
            text: Document text to chunk
            source: Source document path
 
        Returns:
            ChunkResult with sentence-aligned chunks
        """
        if not text or not text.strip():
            return ChunkResult(source=source, chunker_name=self.name)
 
        sentences = split_sentences(text)
        sentences = [
            s for s in sentences
            if len(s.split()) >= self._config.min_sentence_words
        ]
 
        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_words: int = 0
        char_offset: int = 0
 
        for sentence in sentences:
            sentence_words = len(sentence.split())
 
            # If adding this sentence would exceed limit, flush current chunk
            if (
                current_words + sentence_words > self.config.chunk_size
                and current_sentences
            ):
                chunk_text = " ".join(current_sentences)
                chunk = self._make_chunk(chunk_text, len(chunks), char_offset, source)
                if self._is_valid(chunk):
                    chunks.append(chunk)
                char_offset += len(chunk_text) + 1
 
                # Carry over sentence_overlap sentences
                overlap = self._config.sentence_overlap
                current_sentences = current_sentences[-overlap:] if overlap else []
                current_words = sum(len(s.split()) for s in current_sentences)
 
            current_sentences.append(sentence)
            current_words += sentence_words
 
        # Flush remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = self._make_chunk(chunk_text, len(chunks), char_offset, source)
            if self._is_valid(chunk):
                chunks.append(chunk)
 
        logger.debug(
            "%s: %d sentences -> %d chunks (source=%s)",
            self.name, len(sentences), len(chunks), source
        )
 
        return ChunkResult(
            source=source,
            chunks=chunks,
            chunker_name=self.name,
            config_summary=f"size={self.config.chunk_size},overlap={self._config.sentence_overlap}sentences",
        )
 
 
@lru_cache(maxsize=1)
def get_sentence_splitter() -> SentenceSplitter:
    """Factory function - returns cached SentenceSplitter."""
    return SentenceSplitter()
