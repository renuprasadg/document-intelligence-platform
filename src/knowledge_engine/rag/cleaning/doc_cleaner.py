"""
Document cleaning pipeline orchestrator (Enterprise Edition)
 
Correct step order (page boundaries must be intact for header/footer detection):
  1. PDF text extraction (PyMuPDF) — full_text has \x0c page delimiters
  2. Header/footer removal  — runs FIRST, needs page delimiters to detect repeats
  3. Text normalization     — now safe to collapse whitespace
  4. Line wrap repair       — de-hyphenate and unwrap soft breaks
  5. Quality validation     — score the final clean text
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional
 
from knowledge_engine.rag.cleaning.document_quality_validator import (
    QualityReport, get_quality_validator,
)
from knowledge_engine.rag.cleaning.header_footer_filter import (
    FilterResult, get_header_footer_filter,
)
from knowledge_engine.rag.cleaning.line_wrapper import LineWrapResult, get_line_wrapper
from knowledge_engine.rag.cleaning.pdf_text_extractor import (
    ExtractionResult, get_pdf_extractor,
)
from knowledge_engine.rag.cleaning.text_normalizer import (
    NormalizationResult, get_text_normalizer,
)
from knowledge_engine.core.exceptions import DocumentProcessingError
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
@dataclass
class CleanResult:
    """Complete result of the document cleaning pipeline"""
    source_path: str
    clean_text: str
    extraction: Optional[ExtractionResult] = None
    normalization: Optional[NormalizationResult] = None
    line_wrap: Optional[LineWrapResult] = None
    filter_result: Optional[FilterResult] = None
    quality: Optional[QualityReport] = None
    pipeline_error: Optional[str] = None
 
    @property
    def succeeded(self) -> bool:
        return self.pipeline_error is None and bool(self.clean_text)
 
    @property
    def quality_passed(self) -> bool:
        return self.quality is not None and self.quality.passed
 
    def summary(self) -> str:
        if not self.succeeded:
            return f"FAILED: {self.pipeline_error}"
        words = len(self.clean_text.split())
        q = self.quality.summary() if self.quality else "quality=unknown"
        return f"OK | words={words} | {q}"
 
 
class DocumentCleaner:
    """
    Orchestrates the full document cleaning pipeline.
 
    Usage:
        cleaner = get_document_cleaner()
        result = cleaner.clean("path/to/doc.pdf")
        if result.quality_passed:
            text = result.clean_text
    """
 
    def __init__(self):
        self._extractor  = get_pdf_extractor()
        self._normalizer = get_text_normalizer()
        self._wrapper    = get_line_wrapper()
        self._hf_filter  = get_header_footer_filter()
        self._validator  = get_quality_validator()
 
    def clean(self, filepath: str) -> CleanResult:
        """
        Run full cleaning pipeline on a PDF file.
 
        Args:
            filepath: Path to PDF file
 
        Returns:
            CleanResult with clean text and pipeline metadata
        """
        logger.info("Starting clean pipeline for: %s", filepath)
 
        # Step 1: Extract
        extraction = self._extractor.extract(filepath)
        if not extraction.succeeded:
            return CleanResult(
                source_path=filepath,
                clean_text="",
                extraction=extraction,
                pipeline_error=extraction.extraction_error,
            )
 
        if extraction.is_scanned:
            logger.warning("Scanned PDF - text quality will be low: %s", filepath)
 
        # full_text preserves form-feed (\x0c) page delimiters from extraction.
        # IMPORTANT: header/footer filter runs FIRST on the page-delimited text
        # so it can see page boundaries and detect repeating patterns correctly.
        # Only after filtering do we normalize and join pages.
        raw_text = extraction.full_text  # contains \x0c page breaks
 
        # Step 2: Filter headers/footers FIRST (needs page boundaries intact)
        filter_result = self._hf_filter.filter(raw_text)
        filtered_text = filter_result.filtered_text
 
        # Step 3: Normalize (now safe to collapse whitespace across pages)
        norm_result = self._normalizer.normalize(filtered_text)
        normalized_text = norm_result.normalized_text
 
        # Step 4: Repair line wraps
        wrap_result = self._wrapper.repair(normalized_text)
        wrapped_text = wrap_result.repaired_text
 
        # Step 5: Validate quality
        quality = self._validator.validate(wrapped_text)
 
        logger.info(
            "Pipeline complete for %s: %s",
            filepath,
            quality.summary(),
        )
 
        return CleanResult(
            source_path=filepath,
            clean_text=wrap_result.repaired_text,
            extraction=extraction,
            normalization=norm_result,
            line_wrap=wrap_result,
            filter_result=filter_result,
            quality=quality,
        )
 
    def clean_text(self, text: str) -> CleanResult:
        """
        Run cleaning pipeline on raw text (skip extraction step).
        Useful for testing or pre-extracted text that still has page delimiters.
        """
        # Mirror the same order as clean(): filter headers first, then normalize
        filter_result = self._hf_filter.filter(text)
        norm_result   = self._normalizer.normalize(filter_result.filtered_text)
        wrap_result   = self._wrapper.repair(norm_result.normalized_text)
        quality       = self._validator.validate(wrap_result.repaired_text)
 
        return CleanResult(
            source_path="<raw_text>",
            clean_text=wrap_result.repaired_text,
            normalization=norm_result,
            line_wrap=wrap_result,
            filter_result=filter_result,
            quality=quality,
        )
 
 
@lru_cache(maxsize=1)
def get_document_cleaner() -> DocumentCleaner:
    """Factory function - returns cached DocumentCleaner."""
    return DocumentCleaner()
