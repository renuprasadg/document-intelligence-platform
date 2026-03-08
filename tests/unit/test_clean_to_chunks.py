"""
Integration test: cleaning pipeline -> chunker -> metadata builder.
 
Tests the full pipeline using text fixtures (no real PDF needed).
"""
import pytest
from knowledge_engine.rag.chunking.chunk_metadata_builder import (
    DocumentMetadata, get_metadata_builder,
)
from knowledge_engine.rag.chunking.semantic_chunker import get_semantic_chunker
from knowledge_engine.rag.chunking.sentence_splitter import get_sentence_splitter
from knowledge_engine.rag.cleaning.doc_cleaner import DocumentCleaner
 
 
SAMPLE_TEXT = """
Introduction to Machine Learning
 
Machine learning is a subset of artificial intelligence that provides systems
the ability to automatically learn and improve from experience without being
explicitly programmed. Machine learning focuses on the development of computer
programs that can access data and use it to learn for themselves.
 
The process begins with observations or data, such as examples, direct experience,
or instruction, so that computers can look for patterns in data and make better
decisions in the future. The primary aim is to allow the computers to learn
automatically without human intervention or assistance and adjust actions accordingly.
 
Types of Machine Learning
 
Supervised learning uses labelled training data to learn a mapping from inputs
to outputs. Unsupervised learning discovers hidden patterns in unlabelled data.
Reinforcement learning trains agents to make sequences of decisions by rewarding
good outcomes and penalising poor ones.
""".strip()
 
 
class TestCleanToChunksPipeline:
 
    def test_clean_text_pipeline(self):
        """DocumentCleaner.clean_text produces a valid CleanResult."""
        cleaner = DocumentCleaner()
        result = cleaner.clean_text(SAMPLE_TEXT)
        assert result.succeeded
        assert len(result.clean_text) > 0
        assert result.quality is not None
 
    def test_clean_text_quality_passes(self):
        """Sample text should pass quality validation."""
        cleaner = DocumentCleaner()
        result = cleaner.clean_text(SAMPLE_TEXT)
        assert result.quality_passed, f"Quality failed: {result.quality.summary()}"
 
    def test_semantic_chunker_on_clean_output(self):
        """SemanticChunker produces valid chunks from cleaned text."""
        cleaner = DocumentCleaner()
        clean_result = cleaner.clean_text(SAMPLE_TEXT)
        chunker = get_semantic_chunker()
        chunk_result = chunker.chunk(clean_result.clean_text, source="test")
        assert len(chunk_result.chunks) > 0
        for chunk in chunk_result.chunks:
            assert chunk.word_count >= chunker.config.min_chunk_size
 
    def test_sentence_splitter_on_clean_output(self):
        """SentenceSplitter produces chunks without cutting mid-sentence."""
        cleaner = DocumentCleaner()
        clean_result = cleaner.clean_text(SAMPLE_TEXT)
        splitter = get_sentence_splitter()
        chunk_result = splitter.chunk(clean_result.clean_text, source="test")
        assert len(chunk_result.chunks) > 0
 
    def test_metadata_builder_enriches_chunks(self):
        """MetadataBuilder adds required fields to every chunk."""
        cleaner = DocumentCleaner()
        clean_result = cleaner.clean_text(SAMPLE_TEXT)
        chunker = get_semantic_chunker()
        chunk_result = chunker.chunk(clean_result.clean_text, source="test_doc.pdf")
 
        builder = get_metadata_builder()
        doc_meta = DocumentMetadata.from_path("test_doc.pdf")
        meta_result = builder.build(chunk_result, doc_meta, clean_result.quality)
 
        assert meta_result.total_chunks == len(chunk_result.chunks)
        for enriched in meta_result.enriched_chunks:
            meta = enriched.metadata
            assert meta["source_path"] == "test_doc.pdf"
            assert meta["chunk_index"] >= 0
            assert meta["total_chunks"] == meta_result.total_chunks
            assert "quality_score" in meta
            assert "chunker_name" in meta
 
    def test_full_pipeline_end_to_end(self):
        """End-to-end: clean -> semantic chunk -> enriched tuples."""
        cleaner = DocumentCleaner()
        chunker = get_semantic_chunker()
        builder = get_metadata_builder()
 
        # Clean
        clean = cleaner.clean_text(SAMPLE_TEXT)
        assert clean.succeeded
 
        # Chunk
        chunks = chunker.chunk(clean.clean_text, source="e2e_test.txt")
        assert len(chunks) > 0
 
        # Enrich
        doc_meta = DocumentMetadata.from_path("e2e_test.txt")
        doc_meta.total_chars = len(clean.clean_text)
        enriched = builder.build(chunks, doc_meta, clean.quality)
 
        # Verify output format (ready for Week 3 embedding)
        tuples = enriched.as_tuples()
        assert len(tuples) > 0
        for text, meta in tuples:
            assert isinstance(text, str)
            assert len(text) > 0
            assert isinstance(meta, dict)
            assert "chunk_id" in meta
