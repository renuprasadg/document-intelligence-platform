"""
Integration tests for ChromaRepository (Week 3)
 
Tests the ChromaDB vector store layer independently of the embedding layer.
 
Strategy:
  - All tests use a tmp_path-backed ChromaRepository so no shared state leaks.
  - Vectors are tiny (8-dim) deterministic floats — no real embeddings needed.
  - Tests cover: add, query, delete, upsert idempotency, score filtering,
    metadata filtering, per-document cap, and collection stats.
 
Run:
    pytest tests/integration/test_vectorstore.py -v
"""
from __future__ import annotations
 
import math
from pathlib import Path
from typing import Any
 
import pytest
 
from knowledge_engine.rag.vectorstore.chroma_repository import (
    ChunkRecord,
    ChromaRepository,
    QueryResult,
    VectorStoreError,
    make_chunk_id,
)
 
 
# ─── Helpers ──────────────────────────────────────────────────────────────────
 
DIMS = 8
 
 
def _unit_vec(seed: int, dims: int = DIMS) -> list[float]:
    """Deterministic unit vector from an integer seed."""
    raw = [math.sin(seed + i * 1.3) for i in range(dims)]
    mag = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / mag for x in raw]
 
 
def _make_record(
    chunk_id: str,
    text: str,
    embedding: list[float],
    document_id: str = "policy_v1",
    chunk_index: int = 0,
    source_file: str = "data/policy_v1.pdf",
    page_number: int = 1,
    section: str = "Introduction",
    quality_score: float = 0.9,
    chunker_name: str = "SemanticChunker",
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
        document_id=document_id,
        chunk_index=chunk_index,
        source_file=source_file,
        page_number=page_number,
        section=section,
        quality_score=quality_score,
        chunker_name=chunker_name,
        char_start=chunk_index * 200,
        char_end=chunk_index * 200 + len(text),
        word_count=len(text.split()),
    )
 
 
@pytest.fixture()
def repo(tmp_path: Path) -> ChromaRepository:
    """Fresh isolated ChromaRepository for each test."""
    return ChromaRepository(
        collection_name="test_vectorstore",
        persist_dir=str(tmp_path / "chroma"),
    )
 
 
# ─── ChunkRecord ──────────────────────────────────────────────────────────────
 
 
class TestChunkRecord:
    def test_to_chroma_metadata_has_required_fields(self) -> None:
        record = _make_record("id1", "text", _unit_vec(1))
        meta = record.to_chroma_metadata()
 
        required = [
            "document_id", "chunk_index", "source_file",
            "page_number", "section", "char_start", "char_end",
            "word_count", "quality_score", "chunker_name",
        ]
        for field in required:
            assert field in meta, f"Missing metadata field: {field!r}"
 
    def test_to_chroma_metadata_all_safe_types(self) -> None:
        record = _make_record("id1", "some text content", _unit_vec(2))
        meta = record.to_chroma_metadata()
        for k, v in meta.items():
            assert isinstance(v, (str, int, float, bool)), (
                f"Key {k!r} has type {type(v).__name__}"
            )
 
    def test_to_chroma_metadata_coerces_extra_metadata(self) -> None:
        record = _make_record("id1", "text", _unit_vec(3))
        record.extra_metadata = {"a_list": [1, 2, 3], "a_str": "ok"}
        meta = record.to_chroma_metadata()
        assert isinstance(meta["a_list"], str)   # coerced
        assert meta["a_str"] == "ok"
 
 
# ─── Add & Count ─────────────────────────────────────────────────────────────
 
 
class TestAddChunks:
    def test_add_single_chunk(self, repo: ChromaRepository) -> None:
        record = _make_record("c1", "Policy coverage details.", _unit_vec(10))
        stored = repo.add_chunks([record])
        assert stored == 1
        assert repo.count() == 1
 
    def test_add_multiple_chunks(self, repo: ChromaRepository) -> None:
        records = [
            _make_record(f"c{i}", f"Chunk text {i}", _unit_vec(i), chunk_index=i)
            for i in range(5)
        ]
        stored = repo.add_chunks(records)
        assert stored == 5
        assert repo.count() == 5
 
    def test_add_empty_list_returns_zero(self, repo: ChromaRepository) -> None:
        stored = repo.add_chunks([])
        assert stored == 0
        assert repo.count() == 0
 
    def test_upsert_is_idempotent(self, repo: ChromaRepository) -> None:
        """Re-adding the same chunk_id must not grow the collection."""
        record = _make_record("same-id", "original text", _unit_vec(20))
        repo.add_chunks([record])
        assert repo.count() == 1
 
        # Upsert with different text — count must stay 1
        updated = _make_record("same-id", "updated text", _unit_vec(21))
        repo.add_chunks([updated])
        assert repo.count() == 1
 
    def test_batching_stores_all_records(self, repo: ChromaRepository) -> None:
        records = [
            _make_record(f"batch_{i}", f"text {i}", _unit_vec(i), chunk_index=i)
            for i in range(25)
        ]
        stored = repo.add_chunks(records, batch_size=7)
        assert stored == 25
        assert repo.count() == 25
 
 
# ─── Query ───────────────────────────────────────────────────────────────────
 
 
class TestQuery:
    def _populate(self, repo: ChromaRepository) -> None:
        records = [
            _make_record(f"doc_c{i}", f"chunk {i}", _unit_vec(i), chunk_index=i)
            for i in range(10)
        ]
        repo.add_chunks(records)
 
    def test_query_returns_top_k_results(self, repo: ChromaRepository) -> None:
        self._populate(repo)
        results = repo.query(_unit_vec(0), top_k=3)
        assert len(results) == 3
 
    def test_query_results_sorted_by_score_descending(self, repo: ChromaRepository) -> None:
        self._populate(repo)
        results = repo.query(_unit_vec(0), top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
 
    def test_query_score_in_range(self, repo: ChromaRepository) -> None:
        self._populate(repo)
        results = repo.query(_unit_vec(0), top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0
 
    def test_query_returns_query_result_objects(self, repo: ChromaRepository) -> None:
        self._populate(repo)
        results = repo.query(_unit_vec(0), top_k=3)
        for r in results:
            assert isinstance(r, QueryResult)
            assert r.chunk_id
            assert r.text
            assert isinstance(r.metadata, dict)
 
    def test_score_threshold_filters_low_scores(self, repo: ChromaRepository) -> None:
        """Results with score < threshold must be excluded."""
        self._populate(repo)
        results = repo.query(_unit_vec(0), top_k=10, score_threshold=0.999)
        # At threshold=0.999, only near-identical vectors survive
        for r in results:
            assert r.score >= 0.999
 
    def test_query_empty_collection_returns_empty_list(self, repo: ChromaRepository) -> None:
        results = repo.query(_unit_vec(0), top_k=5)
        assert results == []
 
    def test_query_top_k_larger_than_collection(self, repo: ChromaRepository) -> None:
        records = [_make_record(f"x{i}", f"t{i}", _unit_vec(i)) for i in range(3)]
        repo.add_chunks(records)
        results = repo.query(_unit_vec(0), top_k=100)
        # Must not raise; returns all 3
        assert len(results) == 3
 
 
# ─── Metadata Filtering ───────────────────────────────────────────────────────
 
 
class TestMetadataFiltering:
    def _populate_two_docs(self, repo: ChromaRepository) -> None:
        doc_a = [
            _make_record(
                f"a_{i}", f"Doc A chunk {i}", _unit_vec(i),
                document_id="doc_a", chunk_index=i, page_number=i + 1,
            )
            for i in range(4)
        ]
        doc_b = [
            _make_record(
                f"b_{i}", f"Doc B chunk {i}", _unit_vec(i + 20),
                document_id="doc_b", chunk_index=i,
            )
            for i in range(3)
        ]
        repo.add_chunks(doc_a + doc_b)
 
    def test_filter_by_document_id(self, repo: ChromaRepository) -> None:
        self._populate_two_docs(repo)
        where = {"document_id": {"$eq": "doc_a"}}
        results = repo.query(_unit_vec(0), top_k=10, where=where)
        for r in results:
            assert r.metadata["document_id"] == "doc_a"
 
    def test_filter_by_page_number(self, repo: ChromaRepository) -> None:
        self._populate_two_docs(repo)
        where = {"page_number": {"$eq": 2}}
        results = repo.query(_unit_vec(0), top_k=10, where=where)
        for r in results:
            assert r.metadata["page_number"] == 2
 
    def test_filter_by_page_range(self, repo: ChromaRepository) -> None:
        self._populate_two_docs(repo)
        where = {"$and": [
            {"page_number": {"$gte": 2}},
            {"page_number": {"$lte": 3}},
        ]}
        results = repo.query(_unit_vec(0), top_k=10, where=where)
        for r in results:
            assert 2 <= r.metadata["page_number"] <= 3
 
    def test_no_filter_returns_all_docs(self, repo: ChromaRepository) -> None:
        self._populate_two_docs(repo)
        results = repo.query(_unit_vec(0), top_k=10)
        doc_ids = {r.metadata["document_id"] for r in results}
        assert "doc_a" in doc_ids
        assert "doc_b" in doc_ids
 
 
# ─── Delete ──────────────────────────────────────────────────────────────────
 
 
class TestDelete:
    def test_delete_document_removes_its_chunks(self, repo: ChromaRepository) -> None:
        records = [
            _make_record(f"del_{i}", f"text {i}", _unit_vec(i), document_id="to_delete")
            for i in range(3)
        ]
        other = _make_record("keep_1", "keep this", _unit_vec(99), document_id="keep_me")
        repo.add_chunks(records + [other])
        assert repo.count() == 4
 
        repo.delete_document("to_delete")
        assert repo.count() == 1
 
    def test_delete_nonexistent_document_does_not_raise(
        self, repo: ChromaRepository
    ) -> None:
        repo.delete_document("nonexistent_doc")  # must not raise
 
    def test_clear_collection_empties_all_records(self, repo: ChromaRepository) -> None:
        records = [_make_record(f"c{i}", f"t{i}", _unit_vec(i)) for i in range(5)]
        repo.add_chunks(records)
        assert repo.count() == 5
 
        repo.clear_collection()
        assert repo.count() == 0
 
 
# ─── Get by IDs ───────────────────────────────────────────────────────────────
 
 
class TestGetByIds:
    def test_get_existing_chunks(self, repo: ChromaRepository) -> None:
        records = [_make_record(f"gid_{i}", f"text {i}", _unit_vec(i)) for i in range(3)]
        repo.add_chunks(records)
 
        results = repo.get_by_ids(["gid_0", "gid_2"])
        assert len(results) == 2
        ids = {r.chunk_id for r in results}
        assert ids == {"gid_0", "gid_2"}
 
    def test_get_by_ids_score_is_one(self, repo: ChromaRepository) -> None:
        records = [_make_record("g1", "text", _unit_vec(1))]
        repo.add_chunks(records)
        results = repo.get_by_ids(["g1"])
        assert results[0].score == 1.0
 
 
# ─── Stats ────────────────────────────────────────────────────────────────────
 
 
class TestStats:
    def test_stats_returns_expected_keys(self, repo: ChromaRepository) -> None:
        s = repo.stats()
        assert "collection" in s
        assert "persist_dir" in s
        assert "total_chunks" in s
 
    def test_count_reflects_adds(self, repo: ChromaRepository) -> None:
        assert repo.count() == 0
        records = [_make_record(f"s{i}", f"t{i}", _unit_vec(i)) for i in range(4)]
        repo.add_chunks(records)
        assert repo.count() == 4
 
 
# ─── make_chunk_id ────────────────────────────────────────────────────────────
 
 
class TestMakeChunkId:
    def test_deterministic(self) -> None:
        a = make_chunk_id("policy", 3)
        b = make_chunk_id("policy", 3)
        assert a == b
 
    def test_different_inputs_produce_different_ids(self) -> None:
        assert make_chunk_id("policy", 0) != make_chunk_id("policy", 1)
        assert make_chunk_id("policy", 0) != make_chunk_id("claims", 0)
 
    def test_returns_string(self) -> None:
        assert isinstance(make_chunk_id("doc", 0), str)
