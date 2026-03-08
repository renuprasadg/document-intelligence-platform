"""Unit tests for text_normalizer.py"""
import pytest
from knowledge_engine.rag.cleaning.text_normalizer import (
    TextNormalizer, TextNormalizerConfig, NormalizationResult, get_text_normalizer
)
 
 
class TestTextNormalizer:
 
    def setup_method(self):
        self.normalizer = TextNormalizer()
 
    def test_empty_string_returns_empty(self):
        result = self.normalizer.normalize("")
        assert result.normalized_text == ""
        assert not result.was_modified
 
    def test_ligature_replacement(self):
        text = "effi\ufb03cient"
        result = self.normalizer.normalize(text)
        assert "\ufb03" not in result.normalized_text
        assert "ffi" in result.normalized_text
        assert "ligatures" in result.changes_made
 
    def test_soft_hyphen_removal(self):
        text = "doc\u00adument"
        result = self.normalizer.normalize(text)
        assert "\u00ad" not in result.normalized_text
        assert "soft_hyphens" in result.changes_made
 
    def test_smart_quotes_normalized(self):
        text = "\u201cHello\u201d and \u2018world\u2019"
        result = self.normalizer.normalize(text)
        assert '"' in result.normalized_text
        assert "quotes" in result.changes_made
 
    def test_em_dash_normalized(self):
        text = "one\u2014two"
        result = self.normalizer.normalize(text)
        assert "--" in result.normalized_text
        assert "dashes" in result.changes_made
 
    def test_control_chars_removed(self):
        text = "hello\x00world\x01"
        result = self.normalizer.normalize(text)
        assert "\x00" not in result.normalized_text
        assert "\x01" not in result.normalized_text
 
    def test_excess_whitespace_collapsed(self):
        text = "  hello   world  "
        result = self.normalizer.normalize(text)
        assert result.normalized_text == "hello world"
 
    def test_was_modified_false_for_clean_text(self):
        text = "The quick brown fox."
        result = self.normalizer.normalize(text)
        assert not result.was_modified
 
    def test_char_counts_populated(self):
        text = "hello world"
        result = self.normalizer.normalize(text)
        assert result.char_count_before == 11
        assert result.char_count_after > 0
 
    def test_normalize_text_convenience(self):
        text = "  spaces  "
        assert self.normalizer.normalize_text(text) == "spaces"
 
    def test_factory_returns_cached_instance(self):
        a = get_text_normalizer()
        b = get_text_normalizer()
        assert a is b
