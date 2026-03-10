"""
Integration tests for the IndexManager (Week 3)
 
Tests the full pipeline:
    JSONL chunk files → embed → store in ChromaDB
 
Strategy:
  - A FakeEmbeddingProvider replaces the real OpenAI client so no API key
    or network access is required.
  - A tmp_path ChromaDB instance is used for full isolation between tests.
  - The JSONL fixture mirrors the exact Week 2 chunk_documents_cli.py output
    schema so these tests remain valid as Week 2 evolves.
 
Run:
    pytest tests/integration/test_index_build.py -v
"""
from __future__ import annotations
 
import json
import math
from pathlib import Path
from typing import Any
 
import pytest
 
from knowledge_engine.rag.embeddings.embedding_provider import (
    BatchEmbeddingResult,
    EmbeddingProvider,
    EmbeddingResult,
)
from knowledge_engine.rag.vectorstore.chroma_repository import ChromaRepository
from knowledge_engine.rag.vectorstore.index_manager import (
    IndexManager,
    IndexManagerConfig,
    IndexStats,
    _build_chunk_record,
    _iter_jsonl,
)
 
 
# ─── Fixtures & Helpers ───────────────────────────────────────────────────────
 
 
EMBEDDING_DIMS = 8  # tiny dimension for tests
 
 
class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic fake embedding provider.
 
    Produces a unit vector where the first element encodes the text length
    so different texts produce meaningfully different (but cheap) vectors.
    """
 
    def __init__(self, dims: int = EMBEDDING_DIMS) -> None:
        self._dims = dims
        self.call_count = 0
        self.total_texts_embedded = 0
 
    @property
    def model_name(self) -> str:
        return "fake-embedding-model"
 
    @property
    def dimensions(self) -> int:
        return self._dims
 
    def embed_text(self, text: str) -> EmbeddingResult:
        batch = self.embed_batch([text])
        return batch.results[0]
 
    def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        self.call_count += 1
        self.total_texts_embedded += len(texts)
        results = [
            EmbeddingResult(
                text=t,
                embedding=_fake_vector(t, self._dims),
                model=self.model_name,
                token_count=len(t.split()),
            )
            for t in texts
        ]
        return BatchEmbeddingResult(
            results=results,
            model=self.model_name,
            total_tokens=sum(len(t.split()) for t in texts),
        )
 
 
def _fake_vector(text: str, dims: int) -> list[float]:
    """Produce a deterministic unit vector from text."""
    raw = [math.sin(hash(text) + i) for i in range(dims)]
    magnitude = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / magnitude for x in raw]
 
 
def _make_chunk_line(
    text: str,
    chunk_index: int,
    source_path: str = "tests/fixtures/policy_sample.pdf",
    page_estimate: int = 1,
    section: str = "Section 1",
    quality_score: float = 0.85,
    chunker_name: str = "SemanticChunker",
    chunk_id: str | None = None,
) -> dict[str, Any]:
    """Build a JSONL chunk dict matching the Week 2 output schema."""
    doc_stem = Path(source_path).stem
    return {
        "text": text,
        "metadata": {
            "chunk_id": chunk_id or f"{doc_stem}_{chunk_index:04d}",
            "source_path": source_path,
            "document_title": doc_stem.replace("_", " ").title(),
            "chunk_index": chunk_index,
            "total_chunks": 10,
            "char_start": chunk_index * 100,
            "char_end": chunk_index * 100 + len(text),
            "page_estimate": page_estimate,
            "section": section,
            "word_count": len(text.split()),
            "quality_score": quality_score,
            "quality_level": "HIGH" if quality_score >= 0.8 else "MEDIUM",
            "chunker_name": chunker_name,
        },
    }
 
 
@pytest.fixture()
def chunks_dir(tmp_path: Path) -> Path:
    """Return a temporary chunks directory with sample JSONL files."""
    d = tmp_path / "chunks"
    d.mkdir()
    return d
 
 
@pytest.fixture()
def repo(tmp_path: Path) -> ChromaRepository:
    """Isolated ChromaRepository backed by a tmp directory."""
    return ChromaRepository(
        collection_name="test_index_build",
        persist_dir=str(tmp_path / "vectorstore"),
    )
 
 
@pytest.fixture()
def provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()
 
 
@pytest.fixture()
def manager(provider: FakeEmbeddingProvider, repo: ChromaRepository) -> IndexManager:
    return IndexManager(
        embedding_provider=provider,
        repository=repo,
        config=IndexManagerConfig(embedding_batch_size=5, store_batch_size=10),
    )
 
 
# ─── JSONL Utilities ──────────────────────────────────────────────────────────
 
 
class TestIterJsonl:
    def test_reads_valid_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"text": "a", "metadata": {}}),
            json.dumps({"text": "b", "metadata": {}}),
        ]
        path.write_text("\n".join(lines))
        result = list(_iter_jsonl(path))
        assert len(result) == 2
        assert result[0]["text"] == "a"
 
    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text('{"text": "x"}\n\n{"text": "y"}\n')
        result = list(_iter_jsonl(path))
        assert len(result) == 2
 
    def test_skips_comment_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text('# header comment\n{"text": "hello"}\n')
        result = list(_iter_jsonl(path))
        assert len(result) == 1
 
    def test_skips_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text('{"text": "ok"}\nNOT_JSON\n{"text": "also ok"}\n')
        result = list(_iter_jsonl(path))
        # Malformed line skipped — 2 good lines returned
        assert len(result) == 2
 
 
# ─── ChunkRecord Builder ──────────────────────────────────────────────────────
 
 
class TestBuildChunkRecord:
    def test_maps_all_week2_fields(self) -> None:
        raw = _make_chunk_line("Hello world", chunk_index=0)
        record = _build_chunk_record(raw, embedding=[0.1] * EMBEDDING_DIMS)
 
        assert record.text == "Hello world"
        assert record.chunk_index == 0
        assert record.source_file == "tests/fixtures/policy_sample.pdf"
        assert record.document_id == "policy_sample"
        assert record.page_number == 1
        assert record.section == "Section 1"
        assert record.quality_score == pytest.approx(0.85)
        assert record.chunker_name == "SemanticChunker"
        assert record.embedding == [0.1] * EMBEDDING_DIMS
 
    def test_uses_provided_chunk_id(self) -> None:
        raw = _make_chunk_line("text", chunk_index=0, chunk_id="sha256-abc123")
        record = _build_chunk_record(raw, embedding=[0.0] * EMBEDDING_DIMS)
        assert record.chunk_id == "sha256-abc123"
 
    def test_generates_chunk_id_when_missing(self) -> None:
        raw = {"text": "text", "metadata": {"chunk_index": 0, "source_path": "doc.pdf"}}
        record = _build_chunk_record(raw, embedding=[0.0] * EMBEDDING_DIMS)
        assert record.chunk_id  # non-empty
        assert isinstance(record.chunk_id, str)
 
    def test_handles_missing_optional_fields(self) -> None:
        """ChunkRecord must not raise on sparse metadata."""
        raw = {"text": "sparse", "metadata": {}}
        record = _build_chunk_record(raw, embedding=[0.0] * EMBEDDING_DIMS)
        assert record.text == "sparse"
        assert record.page_number == -1
        assert record.section == ""
        assert record.quality_score == 0.0
 
    def test_chroma_metadata_all_safe_types(self) -> None:
        """to_chroma_metadata() must return only str/int/float/bool values."""
        raw = _make_chunk_line("data", chunk_index=2)
        record = _build_chunk_record(raw, embedding=[0.1] * EMBEDDING_DIMS)
        meta = record.to_chroma_metadata()
 
        for k, v in meta.items():
            assert isinstance(v, (str, int, float, bool)), (
                f"Metadata key {k!r} has unsafe type {type(v).__name__}"
            )
 
 
# ─── IndexManager — single-file indexing ─────────────────────────────────────
 
 
class TestIndexFile:
    def test_index_single_jsonl_file(
        self,
        chunks_dir: Path,
        manager: IndexManager,
        repo: ChromaRepository,
    ) -> None:
        path = chunks_dir / "policy.jsonl"
        lines = [_make_chunk_line(f"Chunk text {i}", chunk_index=i) for i in range(4)]
        path.write_text("\n".join(json.dumps(l) for l in lines))
 
        stats = manager.index_file(path)
 
        assert stats.files_processed == 1
        assert stats.chunks_read == 4
        assert stats.chunks_embedded == 4
        assert stats.chunks_stored == 4
        assert stats.chunks_skipped == 0
        assert stats.errors == []
        assert repo.count() == 4
 
    def test_skips_empty_text_chunks(
        self,
        chunks_dir: Path,
        manager: IndexManager,
        repo: ChromaRepository,
    ) -> None:
        path = chunks_dir / "mixed.jsonl"
        lines = [
            _make_chunk_line("Real content here", chunk_index=0),
            {"text": "", "metadata": {"chunk_index": 1, "source_path": "doc.pdf"}},
            {"text": "   ", "metadata": {"chunk_index": 2, "source_path": "doc.pdf"}},
            _make_chunk_line("More real content", chunk_index=3),
        ]
        path.write_text("\n".join(json.dumps(l) for l in lines))
 
        stats = manager.index_file(path)
 
        assert stats.chunks_skipped == 2
        assert repo.count() == 2
 
    def test_index_is_idempotent(
        self,
        chunks_dir: Path,
        manager: IndexManager,
        repo: ChromaRepository,
    ) -> None:
        """Re-indexing the same file must not duplicate records."""
        path = chunks_dir / "idempotent.jsonl"
        lines = [_make_chunk_line(f"Text {i}", chunk_index=i) for i in range(3)]
        path.write_text("\n".join(json.dumps(l) for l in lines))
 
        manager.index_file(path)
        assert repo.count() == 3
 
        manager.index_file(path)  # second run
        assert repo.count() == 3  # must not grow
 
 
# ─── IndexManager — directory indexing ───────────────────────────────────────
 
 
class TestIndexDirectory:
    def test_indexes_all_jsonl_files(
        self,
        chunks_dir: Path,
        manager: IndexManager,
        repo: ChromaRepository,
    ) -> None:
        for fname, n in [("policy.jsonl", 3), ("claims.jsonl", 2)]:
            path = chunks_dir / fname
            lines = [
                _make_chunk_line(f"{fname} chunk {i}", chunk_index=i, source_path=fname)
                for i in range(n)
            ]
            path.write_text("\n".join(json.dumps(l) for l in lines))
 
        stats = manager.index_directory(str(chunks_dir))
 
        assert stats.files_processed == 2
        assert stats.chunks_read == 5
        assert stats.chunks_stored == 5
        assert repo.count() == 5
 
    def test_returns_empty_stats_for_empty_directory(
        self,
        chunks_dir: Path,
        manager: IndexManager,
    ) -> None:
        stats = manager.index_directory(str(chunks_dir))
        assert stats.files_processed == 0
        assert stats.chunks_stored == 0
 
    def test_raises_for_missing_directory(self, manager: IndexManager) -> None:
        from knowledge_engine.core.exceptions import DocumentProcessingError
        with pytest.raises(DocumentProcessingError, match="not found"):
            manager.index_directory("/nonexistent/path/chunks")
 
    def test_multiple_files_batched_correctly(
        self,
        chunks_dir: Path,
        provider: FakeEmbeddingProvider,
        manager: IndexManager,
    ) -> None:
        """Verify embed_batch is called, not embed_text one-by-one."""
        for fname in ["a.jsonl", "b.jsonl"]:
            path = chunks_dir / fname
            lines = [
                _make_chunk_line(f"text {i}", chunk_index=i, source_path=fname)
                for i in range(5)
            ]
            path.write_text("\n".join(json.dumps(l) for l in lines))
 
        provider.call_count = 0  # reset
        manager.index_directory(str(chunks_dir))
 
        # With batch_size=5 and 5 chunks per file → 2 batch calls (one per file)
        assert provider.call_count == 2
        assert provider.total_texts_embedded == 10
 
 
# ─── IndexStats ───────────────────────────────────────────────────────────────
 
 
class TestIndexStats:
    def test_success_true_when_stored_and_no_errors(self) -> None:
        s = IndexStats(chunks_stored=1)
        assert s.success is True
 
    def test_success_false_when_errors(self) -> None:
        s = IndexStats(chunks_stored=1, errors=["something failed"])
        assert s.success is False
 
    def test_success_false_when_nothing_stored(self) -> None:
        s = IndexStats(chunks_stored=0)
        assert s.success is False
 
    def test_str_representation(self) -> None:
        s = IndexStats(files_processed=2, chunks_read=10, chunks_stored=10)
        text = str(s)
        assert "files=2" in text
        assert "stored=10" in text
