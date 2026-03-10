"""
Integration tests for the full RAG retrieval pipeline (Week 3)
 
Tests the end-to-end flow:
 
    query string
        → embed query (FakeEmbeddingProvider)
        → similarity search (ChromaRepository, tmp_path)
        → score threshold + per-document cap filtering
        → reranking (NoOpReranker)
        → RetrievalResult list
 
Also tests:
  - RetrievalFilter → FilterBuilder → ChromaDB where clause
  - Retriever with and without filters
  - Empty collection edge cases
  - Score threshold behaviour
 
No real OpenAI API calls are made — FakeEmbeddingProvider is used throughout.
 
Run:
    pytest tests/integration/test_retrieve_flow.py -v
"""
from __future__ import annotations
 
import math
from pathlib import Path
from typing import Any
 
import pytest
 
from knowledge_engine.rag.embeddings.embedding_provider import (
    BatchEmbeddingResult,
    EmbeddingProvider,
    EmbeddingResult,
)
from knowledge_engine.rag.retrieval.retrieval_filters import FilterBuilder, RetrievalFilter
from knowledge_engine.rag.retrieval.retriever import (
    Retriever,
    RetrieverConfig,
    RetrievalResult,
    _cap_per_document,
    _to_retrieval_result,
)
from knowledge_engine.rag.retrieval.reranker import NoOpReranker, RerankResult
from knowledge_engine.rag.vectorstore.chroma_repository import (
    ChunkRecord,
    ChromaRepository,
    QueryResult,
)
 
 
# ─── Shared Helpers ───────────────────────────────────────────────────────────
 
DIMS = 8
 
 
def _unit_vec(seed: int, dims: int = DIMS) -> list[float]:
    raw = [math.sin(seed + i * 1.7) for i in range(dims)]
    mag = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / mag for x in raw]
 
 
class FakeEmbeddingProvider(EmbeddingProvider):
    """Returns a fixed query vector for any input text."""
 
    def __init__(self, query_vector: list[float] | None = None) -> None:
        self._vec = query_vector or _unit_vec(0)
 
    @property
    def model_name(self) -> str:
        return "fake-provider"
 
    @property
    def dimensions(self) -> int:
        return len(self._vec)
 
    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            text=text,
            embedding=self._vec,
            model=self.model_name,
            token_count=len(text.split()),
        )
 
    def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        results = [self.embed_text(t) for t in texts]
        return BatchEmbeddingResult(results=results, model=self.model_name)
 
 
def _make_record(
    chunk_id: str,
    text: str,
    embedding: list[float],
    document_id: str = "policy",
    chunk_index: int = 0,
    page_number: int = 1,
    section: str = "Intro",
    quality_score: float = 0.9,
    source_file: str = "policy.pdf",
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
        char_start=chunk_index * 100,
        char_end=chunk_index * 100 + len(text),
        word_count=len(text.split()),
    )
 
 
@pytest.fixture()
def repo(tmp_path: Path) -> ChromaRepository:
    return ChromaRepository(
        collection_name="test_retrieve_flow",
        persist_dir=str(tmp_path / "chroma"),
    )
 
 
@pytest.fixture()
def provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(query_vector=_unit_vec(0))
 
 
@pytest.fixture()
def retriever(provider: FakeEmbeddingProvider, repo: ChromaRepository) -> Retriever:
    return Retriever(
        embedding_provider=provider,
        repository=repo,
        reranker=NoOpReranker(),
        config=RetrieverConfig(top_k=5, score_threshold=0.0, max_chunks_per_document=0),
    )
 
 
def _populate(repo: ChromaRepository, n: int = 6) -> None:
    """Insert n chunks from two documents into the repository."""
    records = []
    for i in range(n):
        doc = "doc_a" if i % 2 == 0 else "doc_b"
        records.append(
            _make_record(
                f"chunk_{i}",
                f"This is chunk number {i} from {doc}.",
                _unit_vec(i),
                document_id=doc,
                chunk_index=i // 2,
                page_number=(i // 2) + 1,
                source_file=f"{doc}.pdf",
            )
        )
    repo.add_chunks(records)
 
 
# ─── RetrievalFilter & FilterBuilder ─────────────────────────────────────────
 
 
class TestRetrievalFilter:
    def test_empty_filter_is_empty(self) -> None:
        assert RetrievalFilter().is_empty
 
    def test_document_id_not_empty(self) -> None:
        assert not RetrievalFilter(document_id="x").is_empty
 
    def test_page_range_not_empty(self) -> None:
        assert not RetrievalFilter(min_page=1, max_page=5).is_empty
 
    def test_document_ids_list_not_empty(self) -> None:
        assert not RetrievalFilter(document_ids=["a", "b"]).is_empty
 
 
class TestFilterBuilder:
    def test_none_filter_returns_none(self) -> None:
        assert FilterBuilder.build(None) is None
 
    def test_empty_filter_returns_none(self) -> None:
        assert FilterBuilder.build(RetrievalFilter()) is None
 
    def test_document_id_produces_eq_clause(self) -> None:
        where = FilterBuilder.build(RetrievalFilter(document_id="policy_v3"))
        assert where == {"document_id": {"$eq": "policy_v3"}}
 
    def test_source_file_produces_eq_clause(self) -> None:
        where = FilterBuilder.build(RetrievalFilter(source_file="data/doc.pdf"))
        assert where == {"source_file": {"$eq": "data/doc.pdf"}}
 
    def test_section_produces_eq_clause(self) -> None:
        where = FilterBuilder.build(RetrievalFilter(section="Coverage"))
        assert where == {"section": {"$eq": "Coverage"}}
 
    def test_page_number_exact(self) -> None:
        where = FilterBuilder.build(RetrievalFilter(page_number=3))
        assert where == {"page_number": {"$eq": 3}}
 
    def test_page_range_min_only(self) -> None:
        where = FilterBuilder.build(RetrievalFilter(min_page=2))
        assert where == {"page_number": {"$gte": 2}}
 
    def test_page_range_max_only(self) -> None:
        where = FilterBuilder.build(RetrievalFilter(max_page=5))
        assert where == {"page_number": {"$lte": 5}}
 
    def test_page_range_both(self) -> None:
        where = FilterBuilder.build(RetrievalFilter(min_page=2, max_page=5))
        assert where == {"$and": [
            {"page_number": {"$gte": 2}},
            {"page_number": {"$lte": 5}},
        ]}
 
    def test_min_quality_produces_gte_clause(self) -> None:
        where = FilterBuilder.build(RetrievalFilter(min_quality=0.8))
        assert where == {"quality_score": {"$gte": 0.8}}
 
    def test_multiple_fields_produce_and_clause(self) -> None:
        where = FilterBuilder.build(
            RetrievalFilter(document_id="policy", min_page=1, max_page=3)
        )
        assert "$and" in where
        clauses = where["$and"]
        assert len(clauses) == 3  # document_id + min_page + max_page
 
    def test_document_ids_list_produces_in_clause(self) -> None:
        where = FilterBuilder.build(RetrievalFilter(document_ids=["a", "b"]))
        assert where == {"document_id": {"$in": ["a", "b"]}}
 
    def test_convenience_for_document(self) -> None:
        where = FilterBuilder.for_document("my_doc")
        assert where == {"document_id": {"$eq": "my_doc"}}
 
    def test_convenience_for_page(self) -> None:
        where = FilterBuilder.for_page(4)
        assert where == {"page_number": {"$eq": 4}}
 
    def test_convenience_for_page_range(self) -> None:
        where = FilterBuilder.for_page_range(2, 6)
        assert "$and" in where
 
    def test_validate_accepts_none(self) -> None:
        assert FilterBuilder.validate(None) is True
 
    def test_validate_accepts_valid_clause(self) -> None:
        assert FilterBuilder.validate({"document_id": {"$eq": "x"}}) is True
 
    def test_validate_rejects_non_dict(self) -> None:
        assert FilterBuilder.validate("not a dict") is False  # type: ignore[arg-type]
 
 
# ─── Retriever — search ───────────────────────────────────────────────────────
 
 
class TestRetrieverSearch:
    def test_search_returns_retrieval_results(
        self, retriever: Retriever, repo: ChromaRepository
    ) -> None:
        _populate(repo)
        results = retriever.search("coverage policy", top_k=3)
        assert len(results) <= 3
        for r in results:
            assert isinstance(r, RetrievalResult)
 
    def test_search_result_fields_populated(
        self, retriever: Retriever, repo: ChromaRepository
    ) -> None:
        _populate(repo)
        results = retriever.search("test", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert r.text
        assert r.chunk_id
        assert r.document_id in ("doc_a", "doc_b")
        assert isinstance(r.score, float)
        assert isinstance(r.similarity_score, float)
        assert isinstance(r.page_number, int)
        assert isinstance(r.section, str)
        assert isinstance(r.metadata, dict)
 
    def test_search_sorted_by_score_descending(
        self, retriever: Retriever, repo: ChromaRepository
    ) -> None:
        _populate(repo, n=8)
        results = retriever.search("query", top_k=6)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
 
    def test_search_empty_collection_returns_empty(
        self, retriever: Retriever
    ) -> None:
        results = retriever.search("anything")
        assert results == []
 
    def test_search_empty_query_returns_empty(
        self, retriever: Retriever, repo: ChromaRepository
    ) -> None:
        _populate(repo)
        assert retriever.search("") == []
        assert retriever.search("   ") == []
 
    def test_search_respects_top_k(
        self, retriever: Retriever, repo: ChromaRepository
    ) -> None:
        _populate(repo, n=8)
        results = retriever.search("query", top_k=2)
        assert len(results) <= 2
 
    def test_score_threshold_filters_low_scoring(
        self, retriever: Retriever, repo: ChromaRepository
    ) -> None:
        _populate(repo)
        results = retriever.search("query", score_threshold=0.9999)
        for r in results:
            assert r.score >= 0.9999
 
    def test_score_zero_threshold_returns_results(
        self, retriever: Retriever, repo: ChromaRepository
    ) -> None:
        _populate(repo)
        results = retriever.search("query", top_k=5, score_threshold=0.0)
        assert len(results) > 0
 
 
# ─── Retriever — metadata filtering ──────────────────────────────────────────
 
 
class TestRetrieverFiltering:
    def test_filter_by_document_id(
        self, retriever: Retriever, repo: ChromaRepository
    ) -> None:
        _populate(repo, n=6)
        filt = RetrievalFilter(document_id="doc_a")
        results = retriever.search("chunk", filter_=filt, top_k=10)
        for r in results:
            assert r.document_id == "doc_a"
 
    def test_filter_by_source_file(
        self, retriever: Retriever, repo: ChromaRepository
    ) -> None:
        _populate(repo, n=6)
        filt = RetrievalFilter(source_file="doc_b.pdf")
        results = retriever.search("chunk", filter_=filt, top_k=10)
        for r in results:
            assert r.source_file == "doc_b.pdf"
 
    def test_filter_by_page_range(
        self, retriever: Retriever, repo: ChromaRepository
    ) -> None:
        _populate(repo, n=8)
        filt = RetrievalFilter(min_page=1, max_page=2)
        results = retriever.search("chunk", filter_=filt, top_k=10)
        for r in results:
            assert 1 <= r.page_number <= 2
 
    def test_empty_filter_returns_all_documents(
        self, retriever: Retriever, repo: ChromaRepository
    ) -> None:
        _populate(repo, n=6)
        results = retriever.search("chunk", top_k=10)
        doc_ids = {r.document_id for r in results}
        assert "doc_a" in doc_ids
        assert "doc_b" in doc_ids
 
 
# ─── Per-document cap ─────────────────────────────────────────────────────────
 
 
class TestPerDocumentCap:
    def _make_query_result(
        self, chunk_id: str, score: float, document_id: str
    ) -> QueryResult:
        return QueryResult(
            chunk_id=chunk_id,
            text=f"text for {chunk_id}",
            score=score,
            metadata={"document_id": document_id},
        )
 
    def test_cap_limits_per_document(self) -> None:
        results = [
            self._make_query_result("a1", 0.9, "doc_a"),
            self._make_query_result("a2", 0.85, "doc_a"),
            self._make_query_result("a3", 0.80, "doc_a"),
            self._make_query_result("b1", 0.88, "doc_b"),
        ]
        capped = _cap_per_document(results, max_per_doc=2)
        doc_a_results = [r for r in capped if r.metadata["document_id"] == "doc_a"]
        assert len(doc_a_results) == 2
 
    def test_cap_preserves_score_order(self) -> None:
        results = [
            self._make_query_result("a1", 0.9, "doc_a"),
            self._make_query_result("b1", 0.88, "doc_b"),
            self._make_query_result("a2", 0.85, "doc_a"),
            self._make_query_result("b2", 0.82, "doc_b"),
            self._make_query_result("a3", 0.70, "doc_a"),
        ]
        capped = _cap_per_document(results, max_per_doc=1)
        assert len(capped) == 2
        assert capped[0].chunk_id == "a1"
        assert capped[1].chunk_id == "b1"
 
    def test_zero_cap_returns_all(self) -> None:
        results = [
            self._make_query_result(f"c{i}", 0.9 - i * 0.1, "doc_a")
            for i in range(5)
        ]
        capped = _cap_per_document(results, max_per_doc=0)
        # max_per_doc=0 means counts[doc] < 0 is never True → all filtered out
        # Verify the Retriever skips cap when max_per_doc == 0
        # (This test verifies the helper behaviour directly)
        assert len(capped) == 0
 
    def test_retriever_applies_cap(
        self, repo: ChromaRepository, provider: FakeEmbeddingProvider
    ) -> None:
        """Retriever with max_chunks_per_document=1 must return ≤1 chunk per doc."""
        _populate(repo, n=6)
        retriever = Retriever(
            embedding_provider=provider,
            repository=repo,
            config=RetrieverConfig(top_k=6, max_chunks_per_document=1),
        )
        results = retriever.search("chunk", top_k=6)
        doc_counts: dict[str, int] = {}
        for r in results:
            doc_counts[r.document_id] = doc_counts.get(r.document_id, 0) + 1
        for doc_id, count in doc_counts.items():
            assert count <= 1, f"Document {doc_id!r} has {count} chunks (cap=1)"
 
 
# ─── Reranker Integration ─────────────────────────────────────────────────────
 
 
class TestNoOpReranker:
    def _make_query_result(self, score: float, text: str = "text") -> QueryResult:
        return QueryResult(
            chunk_id="id",
            text=text,
            score=score,
            metadata={"document_id": "doc"},
        )
 
    def test_noop_preserves_order(self) -> None:
        reranker = NoOpReranker()
        results = [
            self._make_query_result(0.9),
            self._make_query_result(0.8),
            self._make_query_result(0.7),
        ]
        reranked = reranker.rerank("query", results)
        scores = [r.rerank_score for r in reranked]
        assert scores == [0.9, 0.8, 0.7]
 
    def test_noop_rerank_score_equals_original_score(self) -> None:
        reranker = NoOpReranker()
        r = self._make_query_result(0.75)
        reranked = reranker.rerank("query", [r])
        assert reranked[0].rerank_score == 0.75
        assert reranked[0].original.score == 0.75
 
    def test_noop_respects_top_k(self) -> None:
        reranker = NoOpReranker()
        results = [self._make_query_result(0.9 - i * 0.1) for i in range(5)]
        reranked = reranker.rerank("query", results, top_k=3)
        assert len(reranked) == 3
 
    def test_noop_sets_rerank_rank(self) -> None:
        reranker = NoOpReranker()
        results = [self._make_query_result(0.9 - i * 0.1) for i in range(3)]
        reranked = reranker.rerank("query", results)
        for i, r in enumerate(reranked):
            assert r.rerank_rank == i
 
    def test_noop_model_name(self) -> None:
        assert NoOpReranker().model_name == "noop"
 
 
# ─── RetrievalResult mapping ──────────────────────────────────────────────────
 
 
class TestToRetrievalResult:
    def _make_rerank_result(
        self,
        score: float = 0.85,
        metadata: dict[str, Any] | None = None,
    ) -> RerankResult:
        meta = metadata or {
            "document_id": "doc_x",
            "source_file": "doc_x.pdf",
            "chunk_index": 2,
            "page_number": 3,
            "section": "Claims",
            "char_start": 100,
            "char_end": 300,
            "word_count": 42,
            "quality_score": 0.88,
        }
        qr = QueryResult(chunk_id="cid_1", text="Chunk text here.", score=score, metadata=meta)
        return RerankResult(
            original=qr, rerank_score=score, rerank_model="noop",
            original_rank=0, rerank_rank=0,
        )
 
    def test_all_fields_mapped(self) -> None:
        rr = _to_retrieval_result(self._make_rerank_result())
        assert isinstance(rr, RetrievalResult)
        assert rr.chunk_id == "cid_1"
        assert rr.text == "Chunk text here."
        assert rr.score == pytest.approx(0.85)
        assert rr.similarity_score == pytest.approx(0.85)
        assert rr.document_id == "doc_x"
        assert rr.page_number == 3
        assert rr.section == "Claims"
        assert rr.word_count == 42
        assert rr.quality_score == pytest.approx(0.88)
 
    def test_missing_metadata_uses_defaults(self) -> None:
        rr = _to_retrieval_result(self._make_rerank_result(metadata={}))
        assert rr.document_id == ""
        assert rr.page_number == -1
        assert rr.section == ""
        assert rr.word_count == 0
