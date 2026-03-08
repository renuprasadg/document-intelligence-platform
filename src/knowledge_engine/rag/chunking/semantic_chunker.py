"""
Semantic/structural chunker for document pipelines (Enterprise Edition)
 
Splits at paragraph and section boundaries first, falling back to
sentence boundaries for oversized paragraphs.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional
 
from knowledge_engine.rag.chunking.base_chunker import (
    BaseChunker, Chunk, ChunkConfig, ChunkResult,
)
from knowledge_engine.rag.chunking.sentence_splitter import split_sentences
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
@dataclass
class SemanticChunkConfig(ChunkConfig):
    """Extended config for SemanticChunker"""
    detect_headers: bool = True
    merge_short_paragraphs: bool = True
    short_para_threshold: int = 20  # Merge paragraphs shorter than N words
 
 
# Section header patterns
_HEADER_PATTERNS = [
    re.compile(r"^#{1,6}\s+.+", re.MULTILINE),          # Markdown ## headers
    re.compile(r"^[A-Z][A-Z\s]{4,}$", re.MULTILINE),    # ALL CAPS HEADING
    re.compile(r"^\d+\.\d*\s+[A-Z]", re.MULTILINE),  # "1.1 Introduction"
    re.compile(r"^(?:Chapter|Section|Part)\s+\d+", re.MULTILINE | re.IGNORECASE),
]
 
 
def _is_section_header(text: str) -> bool:
    """Return True if text looks like a section header."""
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    return any(p.match(stripped) for p in _HEADER_PATTERNS)
 
 
class SemanticChunker(BaseChunker):
    """
    Splits documents at semantic boundaries (sections, paragraphs).
 
    Usage:
        chunker = get_semantic_chunker()
        result = chunker.chunk(clean_text, source="annual_report.pdf")
    """
 
    def __init__(self, config: Optional[SemanticChunkConfig] = None):
        super().__init__(config or SemanticChunkConfig())
        self._config: SemanticChunkConfig = self.config  # type: ignore
 
    @property
    def name(self) -> str:
        return "SemanticChunker"
 
    def chunk(self, text: str, source: str = "") -> ChunkResult:
        """
        Chunk text at paragraph/section boundaries.
 
        Args:
            text: Cleaned document text
            source: Source document path
 
        Returns:
            ChunkResult with semantically-aligned chunks
        """
        if not text or not text.strip():
            return ChunkResult(source=source, chunker_name=self.name)
 
        # Split into paragraphs
        paragraphs = re.split(r"\n{2,}", text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
 
        if self._config.merge_short_paragraphs:
            paragraphs = self._merge_short(paragraphs)
 
        chunks: List[Chunk] = []
        current_parts: List[str] = []
        current_words = 0
        char_offset = 0
 
        for para in paragraphs:
            para_words = len(para.split())
 
            # If paragraph itself exceeds max size, split it
            if para_words > self.config.max_chunk_size:
                # First flush current buffer
                if current_parts:
                    self._flush(current_parts, chunks, char_offset, source)
                    char_offset += sum(len(p) for p in current_parts) + len(current_parts)
                    current_parts = []
                    current_words = 0
                # Then split this oversized paragraph by sentence
                sub_chunks = self._split_large_paragraph(para, len(chunks), char_offset, source)
                chunks.extend(sub_chunks)
                char_offset += len(para) + 1
                continue
 
            # New section header = force flush
            is_header = self._config.detect_headers and _is_section_header(para)
            if is_header and current_parts:
                self._flush(current_parts, chunks, char_offset, source)
                char_offset += sum(len(p) for p in current_parts) + len(current_parts)
                current_parts = []
                current_words = 0
 
            # If adding this paragraph would exceed chunk_size, flush first
            if current_words + para_words > self.config.chunk_size and current_parts:
                self._flush(current_parts, chunks, char_offset, source)
                char_offset += sum(len(p) for p in current_parts) + len(current_parts)
                current_parts = []
                current_words = 0
 
            current_parts.append(para)
            current_words += para_words
 
        # Flush remainder
        if current_parts:
            self._flush(current_parts, chunks, char_offset, source)
 
        logger.debug(
            "%s: %d paragraphs -> %d chunks (source=%s)",
            self.name, len(paragraphs), len(chunks), source
        )
 
        return ChunkResult(
            source=source,
            chunks=chunks,
            chunker_name=self.name,
            config_summary=f"size={self.config.chunk_size},semantic=True",
        )
 
    def _flush(self, parts: List[str], chunks: List[Chunk],
               char_offset: int, source: str) -> None:
        """Flush accumulated paragraphs as a single chunk."""
        chunk_text = "\n\n".join(parts)
        chunk = self._make_chunk(chunk_text, len(chunks), char_offset, source)
        if self._is_valid(chunk):
            chunks.append(chunk)
 
    def _merge_short(self, paragraphs: List[str]) -> List[str]:
        """Merge very short paragraphs into adjacent ones."""
        if not paragraphs:
            return paragraphs
        merged: List[str] = [paragraphs[0]]
        for para in paragraphs[1:]:
            if len(para.split()) < self._config.short_para_threshold:
                merged[-1] = merged[-1] + "\n\n" + para
            else:
                merged.append(para)
        return merged
 
    def _split_large_paragraph(self, para: str, chunk_index: int,
                                char_offset: int, source: str) -> List[Chunk]:
        """Split an oversized paragraph by sentence boundaries."""
        sentences = split_sentences(para)
        chunks: List[Chunk] = []
        current: List[str] = []
        current_words = 0
 
        for sent in sentences:
            sent_words = len(sent.split())
            if current_words + sent_words > self.config.chunk_size and current:
                chunk_text = " ".join(current)
                chunk = self._make_chunk(chunk_text, chunk_index + len(chunks), char_offset, source)
                if self._is_valid(chunk):
                    chunks.append(chunk)
                char_offset += len(chunk_text) + 1
                current = []
                current_words = 0
            current.append(sent)
            current_words += sent_words
 
        if current:
            chunk_text = " ".join(current)
            chunk = self._make_chunk(chunk_text, chunk_index + len(chunks), char_offset, source)
            if self._is_valid(chunk):
                chunks.append(chunk)
 
        return chunks
 
 
@lru_cache(maxsize=1)
def get_semantic_chunker() -> SemanticChunker:
    """Factory function - returns cached SemanticChunker."""
    return SemanticChunker()
