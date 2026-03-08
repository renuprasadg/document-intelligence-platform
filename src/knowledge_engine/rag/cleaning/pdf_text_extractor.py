"""
PDF text extraction using PyMuPDF (Enterprise Edition)
 
Requires: pip install PyMuPDF
 
Features:
  - Per-page extraction with page metadata
  - Scanned PDF detection (image-only PDFs)
  - Automatic normalization via TextNormalizer
  - Configurable extraction flags
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
 
try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError(
        "PyMuPDF is required: pip install PyMuPDF"
    ) from e
 
from knowledge_engine.rag.cleaning.text_normalizer import get_text_normalizer
from knowledge_engine.core.exceptions import DocumentProcessingError
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
@dataclass
class PageData:
    """Data extracted from a single PDF page"""
    page_number: int           # 1-indexed
    raw_text: str
    normalized_text: str
    char_count: int = 0
    word_count: int = 0
    is_empty: bool = False
 
    def __post_init__(self):
        self.char_count = len(self.normalized_text)
        self.word_count = len(self.normalized_text.split())
        self.is_empty = self.word_count == 0
 
 
@dataclass
class ExtractionResult:
    """Result of a full PDF extraction"""
    filepath: str
    pages: List[PageData] = field(default_factory=list)
    total_pages: int = 0
    empty_pages: int = 0
    is_scanned: bool = False
    extraction_error: Optional[str] = None
 
    @property
    def full_text(self) -> str:
        """Combined text from all pages, separated by form-feed."""
        return "\x0c".join(p.normalized_text for p in self.pages)
 
    @property
    def succeeded(self) -> bool:
        return self.extraction_error is None
 
    @property
    def total_words(self) -> int:
        return sum(p.word_count for p in self.pages)
 
 
@dataclass
class ExtractorConfig:
    """Configuration for PDFTextExtractor"""
    # PyMuPDF text extraction flags
    # TEXT_PRESERVE_LIGATURES=1, TEXT_PRESERVE_WHITESPACE=2
    # TEXT_PRESERVE_SPANS=4, TEXT_INHIBIT_SPACES=8
    flags: int = 0
    normalize_text: bool = True
    # Scanned detection: if >= this fraction of pages have fewer than
    # min_page_words words, the PDF likely has no text layer (image-only).
    scanned_page_fraction: float = 0.7   # 70% near-empty pages = scanned
    min_page_words: int = 5              # Pages with fewer words = "empty"
 
 
class PDFTextExtractor:
    """
    Extracts text from PDF files using PyMuPDF.
 
    Usage:
        extractor = get_pdf_extractor()
        result = extractor.extract("path/to/document.pdf")
        if result.is_scanned:
            logger.warning("PDF appears to be scanned - no text layer")
        text = result.full_text
    """
 
    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        self._normalizer = get_text_normalizer() if config is None else None
 
    def extract(self, filepath: str) -> ExtractionResult:
        """
        Extract text from a PDF file.
 
        Args:
            filepath: Path to the PDF file
 
        Returns:
            ExtractionResult with per-page data and full text
 
        Raises:
            DocumentProcessingError: If PDF cannot be opened
        """
        path = Path(filepath)
        if not path.exists():
            raise DocumentProcessingError(f"File not found: {filepath}")
        if path.suffix.lower() != ".pdf":
            raise DocumentProcessingError(f"Not a PDF file: {filepath}")
 
        try:
            return self._extract_pages(filepath)
        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error("PDF extraction failed for %s: %s", filepath, e)
            return ExtractionResult(
                filepath=filepath,
                extraction_error=str(e),
            )
 
    def _extract_pages(self, filepath: str) -> ExtractionResult:
        """Internal: open PDF and extract per-page data."""
        normalizer = self._normalizer or get_text_normalizer()
        pages: List[PageData] = []
 
        with fitz.open(filepath) as doc:
            total_pages = len(doc)
            logger.info("Extracting %d pages from %s", total_pages, filepath)
 
            for page_num in range(total_pages):
                page = doc[page_num]
                raw_text = page.get_text(flags=self.config.flags)
 
                if self.config.normalize_text:
                    norm_result = normalizer.normalize(raw_text)
                    normalized = norm_result.normalized_text
                else:
                    normalized = raw_text
 
                page_data = PageData(
                    page_number=page_num + 1,
                    raw_text=raw_text,
                    normalized_text=normalized,
                )
                pages.append(page_data)
 
        # A page is "empty" if it has fewer than min_page_words words.
        # If 70%+ of pages are empty, the PDF is likely scanned (no text layer).
        empty_pages = sum(
            1 for p in pages if p.word_count < self.config.min_page_words
        )
        is_scanned = (
            total_pages > 0
            and (empty_pages / total_pages) >= self.config.scanned_page_fraction
        )
 
        if is_scanned:
            logger.warning("PDF appears to be scanned (no text layer): %s", filepath)
 
        return ExtractionResult(
            filepath=filepath,
            pages=pages,
            total_pages=total_pages,
            empty_pages=empty_pages,
            is_scanned=is_scanned,
        )
 
 
@lru_cache(maxsize=1)
def get_pdf_extractor() -> PDFTextExtractor:
    """Factory function - returns cached PDFTextExtractor."""
    return PDFTextExtractor()
