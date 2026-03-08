"""
Header/footer detection and removal for PDF-extracted text (Enterprise Edition)
 
Strategy:
  1. Split text into page blocks
  2. Find lines that repeat across >= threshold% of pages
  3. Mark those lines as headers/footers
  4. Remove them from all pages
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Set, Optional
 
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
@dataclass
class HeaderFooterConfig:
    """Configuration for HeaderFooterFilter"""
    page_delimiter: str = "\x0c"      # Form-feed = page break in PyMuPDF
    repeat_threshold: float = 0.4     # Line on 40%+ of pages = header/footer
    header_lines: int = 3             # Check top N lines per page
    footer_lines: int = 3             # Check bottom N lines per page
    min_pages: int = 3                # Need at least N pages to detect patterns
    strip_page_numbers: bool = True
 
 
@dataclass
class FilterResult:
    """Result of header/footer filtering"""
    original_text: str
    filtered_text: str
    headers_removed: List[str] = field(default_factory=list)
    footers_removed: List[str] = field(default_factory=list)
    pages_processed: int = 0
 
    @property
    def patterns_removed(self) -> int:
        return len(self.headers_removed) + len(self.footers_removed)
 
 
# Common page-number patterns
PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),              # Bare number
    re.compile(r"^\s*[Pp]age\s+\d+\s*$"),   # "Page 5"
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),    # "- 5 -"
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"), # "5 / 20"
]
 
 
def _is_page_number(line: str) -> bool:
    """Check if a line is just a page number."""
    return any(p.match(line.strip()) for p in PAGE_NUMBER_PATTERNS)
 
 
class HeaderFooterFilter:
    """
    Detects and removes repeating headers/footers from multi-page text.
 
    Usage:
        hf_filter = get_header_footer_filter()
        result = hf_filter.filter(multi_page_text)
    """
 
    def __init__(self, config: Optional[HeaderFooterConfig] = None):
        self.config = config or HeaderFooterConfig()
 
    def filter(self, text: str) -> FilterResult:
        """
        Detect and remove header/footer lines from extracted text.
 
        Args:
            text: Multi-page text (pages delimited by form-feed or marker)
 
        Returns:
            FilterResult with cleaned text and removed patterns
        """
        pages = text.split(self.config.page_delimiter)
        pages = [p for p in pages if p.strip()]
 
        if len(pages) < self.config.min_pages:
            logger.debug("Too few pages (%d) for header/footer detection", len(pages))
            return FilterResult(
                original_text=text,
                filtered_text=text,
                pages_processed=len(pages),
            )
 
        headers_to_remove = self._detect_patterns(pages, "header")
        footers_to_remove = self._detect_patterns(pages, "footer")
 
        if self.config.strip_page_numbers:
            page_num_lines = self._detect_page_numbers(pages)
            footers_to_remove.update(page_num_lines)
 
        cleaned_pages = [
            self._remove_from_page(page, headers_to_remove, footers_to_remove)
            for page in pages
        ]
 
        filtered_text = "\n\n".join(p for p in cleaned_pages if p.strip())
 
        return FilterResult(
            original_text=text,
            filtered_text=filtered_text,
            headers_removed=list(headers_to_remove),
            footers_removed=list(footers_to_remove),
            pages_processed=len(pages),
        )
 
    def _detect_patterns(self, pages: List[str], location: str) -> Set[str]:
        """Find lines that repeat at header or footer positions."""
        counts: Counter = Counter()
        for page in pages:
            lines = [l.strip() for l in page.strip().split("\n") if l.strip()]
            if not lines:
                continue
            candidates = (
                lines[:self.config.header_lines] if location == "header"
                else lines[-self.config.footer_lines:]
            )
            for candidate in candidates:
                if len(candidate) > 2:  # skip trivially short lines
                    counts[candidate] += 1
 
        threshold_count = len(pages) * self.config.repeat_threshold
        return {line for line, count in counts.items() if count >= threshold_count}
 
    def _detect_page_numbers(self, pages: List[str]) -> Set[str]:
        """Collect all page-number lines across pages."""
        page_nums: Set[str] = set()
        for page in pages:
            for line in page.split("\n"):
                if _is_page_number(line):
                    page_nums.add(line.strip())
        return page_nums
 
    def _remove_from_page(self, page: str, headers: Set[str], footers: Set[str]) -> str:
        """Remove known header/footer lines from a single page."""
        remove = headers | footers
        lines = page.split("\n")
        cleaned = [l for l in lines if l.strip() not in remove]
        return "\n".join(cleaned)
 
 
@lru_cache(maxsize=1)
def get_header_footer_filter() -> HeaderFooterFilter:
    """Factory function - returns cached HeaderFooterFilter."""
    return HeaderFooterFilter()
