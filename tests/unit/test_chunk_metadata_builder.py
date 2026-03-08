"""Unit tests for chunk_metadata_builder.py"""
import pytest
from knowledge_engine.rag.chunking.base_chunker import Chunk, ChunkResult
from knowledge_engine.rag.chunking.chunk_metadata_builder import (
    ChunkMetadataBuilder, DocumentMetadata, get_metadata_builder
)
 
 
def make_chunk_result(n_chunks: int = 3, source: str = "test.pdf") -> ChunkResult:
    """Helper: create a ChunkResult with n simple chunks."""
    chunks = [
        Chunk(
            text=f"This is chunk number {i} with enough words to be valid.",
            chunk_index=i,
            char_start=i * 100,
            char_end=i * 100 + 80,
            source=source,
        )
        for i in range(n_chunks)
    ]
    return ChunkResult(source=source, chunks=chunks, chunker_name="TestChunker")
 
 
class TestChunkMetadataBuilder:
 
    def setup_method(self):
        self.builder = ChunkMetadataBuilder()
 
    def test_builds_correct_chunk_count(self):
        cr = make_chunk_result(4)
        doc_meta = DocumentMetadata.from_path("test.pdf")
        result = self.builder.build(cr, doc_meta)
        assert result.total_chunks == 4
        assert len(result.enriched_chunks) == 4
 
    def test_metadata_contains_required_fields(self):
        cr = make_chunk_result(1)
        doc_meta = DocumentMetadata.from_path("test.pdf")
        result = self.builder.build(cr, doc_meta)
        meta = result.enriched_chunks[0].metadata
        assert "chunk_id" in meta
        assert "source_path" in meta
        assert "chunk_index" in meta
        assert "total_chunks" in meta
        assert "word_count" in meta
        assert "chunker_name" in meta
 
    def test_chunk_id_is_deterministic(self):
        cr = make_chunk_result(1)
        doc_meta = DocumentMetadata.from_path("test.pdf")
        result1 = self.builder.build(cr, doc_meta)
        result2 = self.builder.build(cr, doc_meta)
        assert result1.enriched_chunks[0].chunk_id == result2.enriched_chunks[0].chunk_id
 
    def test_as_tuples_returns_text_and_metadata(self):
        cr = make_chunk_result(2)
        doc_meta = DocumentMetadata.from_path("test.pdf")
        result = self.builder.build(cr, doc_meta)
        tuples = result.as_tuples()
        assert len(tuples) == 2
        text, meta = tuples[0]
        assert isinstance(text, str)
        assert isinstance(meta, dict)
 
    def test_quality_report_propagated(self):
        from knowledge_engine.rag.cleaning.document_quality_validator import (
            QualityReport, QualityLevel
        )
        cr = make_chunk_result(1)
        doc_meta = DocumentMetadata.from_path("test.pdf")
        qr = QualityReport(
            text_length=1000, word_count=200,
            overall_score=0.85, quality_level=QualityLevel.HIGH
        )
        result = self.builder.build(cr, doc_meta, quality_report=qr)
        meta = result.enriched_chunks[0].metadata
        assert meta["quality_score"] == 0.85
        assert meta["quality_level"] == "high"
 
    def test_document_metadata_from_path(self):
        meta = DocumentMetadata.from_path("/data/annual_report.pdf")
        assert meta.document_title == "Annual Report"
 
    def test_factory_returns_cached_instance(self):
        a = get_metadata_builder()
        b = get_metadata_builder()
        assert a is b
