"""Unit tests for header_footer_filter.py"""
import pytest
from knowledge_engine.rag.cleaning.header_footer_filter import (
    HeaderFooterFilter, HeaderFooterConfig, get_header_footer_filter
)
 
 
def make_pages(page_count: int, header: str = "COMPANY CONFIDENTIAL",
               footer: str = "Page 1") -> str:
    """Helper: create multi-page text with consistent header/footer."""
    pages = []
    for i in range(page_count):
        pages.append(
            f"{header}\nContent paragraph for page {i}.\nMore content here.\n{footer}"
        )
    return "\x0c".join(pages)
 
 
class TestHeaderFooterFilter:
 
    def setup_method(self):
        self.filt = HeaderFooterFilter()
 
    def test_detects_repeated_header(self):
        text = make_pages(5, header="CONFIDENTIAL DOCUMENT")
        result = self.filt.filter(text)
        assert "CONFIDENTIAL DOCUMENT" in result.headers_removed
        assert "CONFIDENTIAL DOCUMENT" not in result.filtered_text
 
    def test_detects_page_numbers(self):
        pages = []
        for i in range(5):
            pages.append(f"Content on page {i}.\n{i + 1}")
        text = "\x0c".join(pages)
        result = self.filt.filter(text)
        assert result.patterns_removed > 0
 
    def test_too_few_pages_skips_detection(self):
        text = "Page 1 content\x0cPage 2 content"
        result = self.filt.filter(text)
        assert result.filtered_text == text
 
    def test_pages_processed_count(self):
        text = make_pages(6)
        result = self.filt.filter(text)
        assert result.pages_processed == 6
 
    def test_unique_content_preserved(self):
        text = make_pages(5)
        result = self.filt.filter(text)
        assert "Content paragraph" in result.filtered_text
 
    def test_factory_returns_cached_instance(self):
        a = get_header_footer_filter()
        b = get_header_footer_filter()
        assert a is b
