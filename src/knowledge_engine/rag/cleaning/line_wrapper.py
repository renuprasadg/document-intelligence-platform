"""
Line wrapping repair for PDF-extracted text (Enterprise Edition)
 
Fixes:
  - Hyphenated line breaks (re-joins split words)
  - Soft line wraps within paragraphs
  - Preserves intentional paragraph breaks
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional
 
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
@dataclass
class LineWrapConfig:
    """Configuration for LineWrapper"""
    join_hyphenated: bool = True
    join_soft_wraps: bool = True
    min_line_length: int = 40   # Lines shorter than this may be wrapped
    preserve_bullets: bool = True
 
 
@dataclass
class LineWrapResult:
    """Result of a line-wrapping repair operation"""
    original_text: str
    repaired_text: str
    hyphen_joins: int = 0
    soft_wrap_joins: int = 0
 
    @property
    def total_repairs(self) -> int:
        return self.hyphen_joins + self.soft_wrap_joins
 
 
class LineWrapper:
    """
    Repairs line-break artifacts from PDF text extraction.
 
    Usage:
        wrapper = get_line_wrapper()
        result = wrapper.repair(extracted_text)
    """
 
    # Bullet / list patterns we must NOT join
    # Single-line pattern: lines starting with a bullet/list marker
    # Matches: •, -, *, numbered (1. 2) or lettered (a. b)) lists
    BULLET_PATTERN = re.compile(
        r"^\s*(?:[\u2022\u2023\u25e6\u2043\u2219]|[-*]|\d+[.)]|[a-zA-Z][.)]) ",
        re.MULTILINE
    )
 
    def __init__(self, config: Optional[LineWrapConfig] = None):
        self.config = config or LineWrapConfig()
 
    def repair(self, text: str) -> LineWrapResult:
        """
        Repair line breaks in extracted PDF text.
 
        Args:
            text: Raw text with potential line-break artifacts
 
        Returns:
            LineWrapResult with repaired text and repair counts
        """
        if not text:
            return LineWrapResult(original_text=text, repaired_text=text)
 
        current = text
        hyphen_joins = 0
        soft_wrap_joins = 0
 
        if self.config.join_hyphenated:
            current, hyphen_joins = self._join_hyphenated(current)
 
        if self.config.join_soft_wraps:
            current, soft_wrap_joins = self._join_soft_wraps(current)
 
        return LineWrapResult(
            original_text=text,
            repaired_text=current,
            hyphen_joins=hyphen_joins,
            soft_wrap_joins=soft_wrap_joins,
        )
 
    def _join_hyphenated(self, text: str) -> tuple[str, int]:
        """Join words split by hyphen at line end."""
        # Matches: word- NEWLINE word (not at paragraph break)
        pattern = re.compile(r"([a-zA-Z])-\n([a-zA-Z])")
        count = len(pattern.findall(text))
        repaired = pattern.sub(r"\1\2", text)
        return repaired, count
 
    def _join_soft_wraps(self, text: str) -> tuple[str, int]:
        """Join soft line wraps within paragraphs."""
        lines = text.split("\n")
        result: List[str] = []
        count = 0
 
        for i, line in enumerate(lines):
            stripped = line.rstrip()
 
            # Empty line = paragraph break, preserve
            if not stripped:
                result.append("")
                continue
 
            # Bullet lines - do not join
            if self.config.preserve_bullets and self.BULLET_PATTERN.match(stripped):
                result.append(stripped)
                continue
 
            # Short line followed by non-empty line = soft wrap
            if (
                len(stripped) < self.config.min_line_length
                and i + 1 < len(lines)
                and lines[i + 1].strip()
                and not lines[i + 1].strip().startswith(("\u2022", "-", "*"))
                and not stripped.endswith((".", "!", "?", ":", ";"))
            ):
                result.append(stripped + " ")
                count += 1
            else:
                result.append(stripped)
 
        return "\n".join(result), count
 
    def repair_text(self, text: str) -> str:
        """Convenience method - returns repaired string directly."""
        return self.repair(text).repaired_text
 
 
@lru_cache(maxsize=1)
def get_line_wrapper() -> LineWrapper:
    """Factory function - returns cached LineWrapper instance."""
    return LineWrapper()
