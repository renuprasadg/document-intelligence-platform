"""
Chunk metadata builder for GuardianRAG pipeline (Enterprise Edition)
 
Enriches chunks with provenance metadata needed for vector store indexing.
Returns ChunksWithMetadata - a list of (chunk_text, metadata_dict) pairs
ready for embedding and upsertion.
"""
import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
 
from knowledge_engine.rag.chunking.base_chunker import Chunk, ChunkResult
from knowledge_engine.rag.cleaning.document_quality_validator import QualityReport
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
# Type alias - each item is (text, metadata_dict)
ChunksWithMetadata = List[Tuple[str, Dict[str, Any]]]
 
 
@dataclass
class DocumentMetadata:
    """Document-level metadata shared across all its chunks"""
    source_path: str
    document_title: str = ""
    total_pages: int = 0
    total_chars: int = 0
    quality_score: float = 0.0
    quality_level: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
 
    @classmethod
    def from_path(cls, source_path: str) -> "DocumentMetadata":
        """Create DocumentMetadata from a file path."""
        title = Path(source_path).stem.replace("_", " ").replace("-", " ").title()
        return cls(source_path=source_path, document_title=title)
 
 
@dataclass
class EnrichedChunk:
    """A chunk with full metadata attached"""
    text: str
    metadata: Dict[str, Any]
 
    @property
    def chunk_id(self) -> str:
        return str(self.metadata.get("chunk_id", ""))
 
 
@dataclass
class MetadataResult:
    """Result of metadata enrichment"""
    source: str
    enriched_chunks: List[EnrichedChunk] = field(default_factory=list)
    total_chunks: int = 0
 
    def __post_init__(self):
        if self.total_chunks == 0:
            self.total_chunks = len(self.enriched_chunks)
 
    def as_tuples(self) -> ChunksWithMetadata:
        """Return list of (text, metadata) tuples for embedding."""
        return [(c.text, c.metadata) for c in self.enriched_chunks]
 
    def texts(self) -> List[str]:
        return [c.text for c in self.enriched_chunks]
 
    def metadatas(self) -> List[Dict[str, Any]]:
        return [c.metadata for c in self.enriched_chunks]
 
 
class ChunkMetadataBuilder:
    """
    Enriches raw chunks with full provenance metadata.
 
    Usage:
        builder = get_metadata_builder()
        doc_meta = DocumentMetadata.from_path("report.pdf")
        result = builder.build(chunk_result, doc_meta)
        # result.as_tuples() ready for vector store upsert
    """
 
    def build(
        self,
        chunk_result: ChunkResult,
        doc_meta: DocumentMetadata,
        quality_report: Optional[QualityReport] = None,
    ) -> MetadataResult:
        """
        Build enriched chunks from a ChunkResult.
 
        Args:
            chunk_result: Output from any chunker
            doc_meta: Document-level metadata
            quality_report: Optional quality report from validator
 
        Returns:
            MetadataResult with enriched chunks
        """
        total = len(chunk_result.chunks)
 
        if quality_report:
            doc_meta.quality_score = quality_report.overall_score
            doc_meta.quality_level = quality_report.quality_level.value
 
        enriched: List[EnrichedChunk] = []
 
        for chunk in chunk_result.chunks:
            metadata = self._build_chunk_metadata(
                chunk=chunk,
                total_chunks=total,
                doc_meta=doc_meta,
                chunker_name=chunk_result.chunker_name,
            )
            enriched.append(EnrichedChunk(text=chunk.text, metadata=metadata))
 
        logger.debug(
            "Built metadata for %d chunks from %s",
            len(enriched), doc_meta.source_path
        )
 
        return MetadataResult(
            source=doc_meta.source_path,
            enriched_chunks=enriched,
        )
 
    def _build_chunk_metadata(
        self,
        chunk: Chunk,
        total_chunks: int,
        doc_meta: DocumentMetadata,
        chunker_name: str,
    ) -> Dict[str, Any]:
        """Build metadata dict for a single chunk."""
        # Deterministic chunk ID from source + index
        chunk_id_raw = f"{doc_meta.source_path}::{chunk.chunk_index}"
        chunk_id = hashlib.sha256(chunk_id_raw.encode()).hexdigest()[:16]
 
        # Estimate page number from character position
        page_estimate = self._estimate_page(
            chunk.char_start,
            doc_meta.total_chars,
            doc_meta.total_pages,
        )
 
        return {
            # Provenance
            "chunk_id":       chunk_id,
            "source_path":    doc_meta.source_path,
            "document_title": doc_meta.document_title,
            # Position
            "chunk_index":    chunk.chunk_index,
            "total_chunks":   total_chunks,
            "char_start":     chunk.char_start,
            "char_end":       chunk.char_end,
            "page_estimate":  page_estimate,
            # Content stats
            "word_count":     chunk.word_count,
            "char_count":     len(chunk.text),
            # Quality
            "quality_score":  doc_meta.quality_score,
            "quality_level":  doc_meta.quality_level,
            # Pipeline
            "chunker_name":   chunker_name,
            # Extra (chunker-specific)
            **chunk.extra,
        }
 
    @staticmethod
    def _estimate_page(
        char_start: int,
        total_chars: int,
        total_pages: int,
    ) -> int:
        """Estimate page number from character offset."""
        if total_pages <= 0 or total_chars <= 0:
            return 1
        position_ratio = char_start / total_chars
        return max(1, int(position_ratio * total_pages) + 1)
 
 
@lru_cache(maxsize=1)
def get_metadata_builder() -> ChunkMetadataBuilder:
    """Factory function - returns cached ChunkMetadataBuilder."""
    return ChunkMetadataBuilder()
