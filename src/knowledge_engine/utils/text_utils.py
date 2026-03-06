"""
Text processing utilities
"""
import re
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text"""
    text = text.replace('\t', ' ')
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())


def is_empty_or_whitespace(text: Optional[str]) -> bool:
    """Check if text is None, empty, or only whitespace"""
    return text is None or text.strip() == ""
