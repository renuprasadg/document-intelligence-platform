"""Unit tests for line_wrapper.py"""
import pytest
from knowledge_engine.rag.cleaning.line_wrapper import (
    LineWrapper, LineWrapConfig, get_line_wrapper
)
 
 
class TestLineWrapper:
 
    def setup_method(self):
        self.wrapper = LineWrapper()
 
    def test_joins_hyphenated_line_break(self):
        text = "The iden-\ntical results were expected."
        result = self.wrapper.repair(text)
        assert "iden-\n" not in result.repaired_text
        assert "identical" in result.repaired_text
        assert result.hyphen_joins == 1
 
    def test_does_not_join_bullet_lines(self):
        text = "Header line\n\u2022 First bullet\n\u2022 Second bullet"
        result = self.wrapper.repair(text)
        assert "\u2022 First bullet" in result.repaired_text
        assert "\u2022 Second bullet" in result.repaired_text
 
    def test_empty_text_returns_empty(self):
        result = self.wrapper.repair("")
        assert result.repaired_text == ""
        assert result.total_repairs == 0
 
    def test_total_repairs_property(self):
        text = "hyph-\nens and short\nwrap"
        result = self.wrapper.repair(text)
        assert result.total_repairs >= 0
 
    def test_factory_returns_cached_instance(self):
        a = get_line_wrapper()
        b = get_line_wrapper()
        assert a is b
