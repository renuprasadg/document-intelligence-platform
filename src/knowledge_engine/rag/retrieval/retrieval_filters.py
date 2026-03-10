"""
Retrieval filter builder for the GuardianRAG system (Enterprise Edition)
 
Builds ChromaDB `where` clause dicts from structured filter objects.
 
Design:
  - RetrievalFilter is the public data class callers construct.
  - FilterBuilder.build() converts it to a ChromaDB-compatible where dict.
  - Filters are composable: multiple active fields → $and clause.
 
Supported filter dimensions:
  - document_id    Exact match on document identifier
  - source_file    Exact match on source file path
  - page_number    Exact OR range (min_page / max_page)
  - section        Exact match on section heading string
  - quality_score  Minimum quality threshold (>= value)
  - chunker_name   Exact match on chunker (e.g. "SemanticChunker")
 
Usage:
    from knowledge_engine.rag.retrieval.retrieval_filters import RetrievalFilter, FilterBuilder
 
    # Filter to a specific document, pages 3-7
    f = RetrievalFilter(document_id="policy_v3", min_page=3, max_page=7)
    where = FilterBuilder.build(f)
    # → {"$and": [{"document_id": {"$eq": "policy_v3"}},
    #             {"page_number": {"$gte": 3}},
    #             {"page_number": {"$lte": 7}}]}
 
    results = repo.query(embedding, where=where)
"""
from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Any
 
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
# ─── Filter Dataclass ────────────────────────────────────────────────────────
 
 
@dataclass
class RetrievalFilter:
    """
    Structured retrieval filter.
 
    All fields are optional.  Active (non-None) fields are AND-combined.
 
    Fields:
        document_id:     Filter to a single document (exact match on metadata).
        source_file:     Filter to a specific source file path (exact match).
        section:         Filter to a specific section heading (exact match).
        chunker_name:    Filter to chunks produced by a specific chunker.
        page_number:     Filter to a single exact page number.
        min_page:        Filter to pages >= min_page (inclusive).
        max_page:        Filter to pages <= max_page (inclusive).
        min_quality:     Filter to chunks with quality_score >= this value.
        document_ids:    Filter to any of these document IDs ($in operator).
    """
 
    document_id: str | None = None
    source_file: str | None = None
    section: str | None = None
    chunker_name: str | None = None
    page_number: int | None = None
    min_page: int | None = None
    max_page: int | None = None
    min_quality: float | None = None
    document_ids: list[str] = field(default_factory=list)
 
    @property
    def is_empty(self) -> bool:
        """True when no filter criteria are set (no where clause needed)."""
        return (
            self.document_id is None
            and self.source_file is None
            and self.section is None
            and self.chunker_name is None
            and self.page_number is None
            and self.min_page is None
            and self.max_page is None
            and self.min_quality is None
            and not self.document_ids
        )
 
    def __str__(self) -> str:
        active = {
            k: v
            for k, v in self.__dict__.items()
            if v is not None and v != []
        }
        return f"RetrievalFilter({active})"
 
 
# ─── Filter Builder ──────────────────────────────────────────────────────────
 
 
class FilterBuilder:
    """
    Converts a RetrievalFilter into a ChromaDB `where` clause dict.
 
    ChromaDB where-clause operators:
        $eq    exact equality          {"field": {"$eq": value}}
        $ne    not equal               {"field": {"$ne": value}}
        $gte   greater than or equal   {"field": {"$gte": value}}
        $lte   less than or equal      {"field": {"$lte": value}}
        $gt    greater than            {"field": {"$gt": value}}
        $lt    less than               {"field": {"$lt": value}}
        $in    value in list           {"field": {"$in": [...]}}
        $and   all conditions          {"$and": [{...}, {...}]}
        $or    any condition           {"$or": [{...}, {...}]}
    """
 
    @staticmethod
    def build(filter_: RetrievalFilter | None) -> dict[str, Any] | None:
        """
        Convert a RetrievalFilter to a ChromaDB where clause.
 
        Args:
            filter_: The filter to convert.  None or empty filter → returns None
                     (no where clause applied — full collection searched).
 
        Returns:
            ChromaDB where dict, or None if no filtering required.
        """
        if filter_ is None or filter_.is_empty:
            return None
 
        clauses: list[dict[str, Any]] = []
 
        # ── Exact string matches ──────────────────────────────────────────
        if filter_.document_id is not None:
            clauses.append({"document_id": {"$eq": filter_.document_id}})
 
        if filter_.source_file is not None:
            clauses.append({"source_file": {"$eq": filter_.source_file}})
 
        if filter_.section is not None:
            clauses.append({"section": {"$eq": filter_.section}})
 
        if filter_.chunker_name is not None:
            clauses.append({"chunker_name": {"$eq": filter_.chunker_name}})
 
        # ── Multi-document $in ────────────────────────────────────────────
        if filter_.document_ids:
            clauses.append({"document_id": {"$in": list(filter_.document_ids)}})
 
        # ── Page number ───────────────────────────────────────────────────
        if filter_.page_number is not None:
            # Exact page takes priority over range
            clauses.append({"page_number": {"$eq": int(filter_.page_number)}})
        else:
            if filter_.min_page is not None:
                clauses.append({"page_number": {"$gte": int(filter_.min_page)}})
            if filter_.max_page is not None:
                clauses.append({"page_number": {"$lte": int(filter_.max_page)}})
 
        # ── Quality score ─────────────────────────────────────────────────
        if filter_.min_quality is not None:
            clauses.append({"quality_score": {"$gte": float(filter_.min_quality)}})
 
        # ── Combine ───────────────────────────────────────────────────────
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
 
    @staticmethod
    def for_document(document_id: str) -> dict[str, Any]:
        """Convenience: build a single-document filter directly."""
        return {"document_id": {"$eq": document_id}}
 
    @staticmethod
    def for_page(page_number: int) -> dict[str, Any]:
        """Convenience: build a single-page filter directly."""
        return {"page_number": {"$eq": page_number}}
 
    @staticmethod
    def for_page_range(min_page: int, max_page: int) -> dict[str, Any]:
        """Convenience: build a page-range filter directly."""
        return {"$and": [
            {"page_number": {"$gte": min_page}},
            {"page_number": {"$lte": max_page}},
        ]}
 
    @staticmethod
    def for_section(section: str) -> dict[str, Any]:
        """Convenience: build a section filter directly."""
        return {"section": {"$eq": section}}
 
    @staticmethod
    def for_source_file(source_file: str) -> dict[str, Any]:
        """Convenience: build a source-file filter directly."""
        return {"source_file": {"$eq": source_file}}
 
    @staticmethod
    def validate(where: dict[str, Any] | None) -> bool:
        """
        Lightweight sanity-check on a where clause before sending to ChromaDB.
 
        Checks that:
        - Top-level keys are either field names or $and / $or.
        - Values are dicts (operator clauses) or lists (for $and/$or).
 
        Returns:
            True if the clause looks well-formed, False otherwise.
        """
        if where is None:
            return True
        if not isinstance(where, dict):
            return False
        for key, val in where.items():
            if key in ("$and", "$or"):
                if not isinstance(val, list):
                    return False
            elif not isinstance(val, dict):
                return False
        return True
