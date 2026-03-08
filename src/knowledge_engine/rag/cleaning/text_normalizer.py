"""
Text normalization for GuardianRAG document pipeline (Enterprise Edition)
 
Handles:
  - Unicode normalization (NFC)
  - PDF extraction artifacts (ligatures, soft hyphens)
  - Whitespace standardization
  - Control character removal
"""
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional
 
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
@dataclass
class TextNormalizerConfig:
    """Configuration for TextNormalizer"""
    unicode_form: str = "NFC"          # NFC, NFD, NFKC, NFKD
    fix_ligatures: bool = True
    remove_soft_hyphens: bool = True
    normalize_quotes: bool = True
    normalize_dashes: bool = True
    collapse_whitespace: bool = True
    strip_control_chars: bool = True
 
 
@dataclass
class NormalizationResult:
    """Result of a normalization operation"""
    original_text: str
    normalized_text: str
    changes_made: List[str] = field(default_factory=list)
    char_count_before: int = 0
    char_count_after: int = 0
 
    def __post_init__(self):
        self.char_count_before = len(self.original_text)
        self.char_count_after = len(self.normalized_text)
 
    @property
    def was_modified(self) -> bool:
        return self.original_text != self.normalized_text
 
    @property
    def reduction_pct(self) -> float:
        if self.char_count_before == 0:
            return 0.0
        return (1 - self.char_count_after / self.char_count_before) * 100
 
 
# Ligature map for PDF extraction artifacts
LIGATURE_MAP = {
    "\ufb01": "fi",   # fi ligature
    "\ufb02": "fl",   # fl ligature
    "\ufb03": "ffi",  # ffi ligature
    "\ufb04": "ffl",  # ffl ligature
    "\ufb00": "ff",   # ff ligature
    "\ufb05": "st",   # st ligature
    "\ufb06": "st",   # st ligature (alt)
}
 
 
class TextNormalizer:
    """
    Enterprise text normalizer for document pipeline.
 
    Usage:
        normalizer = get_text_normalizer()
        result = normalizer.normalize(raw_text)
        print(result.normalized_text)
    """
 
    def __init__(self, config: Optional[TextNormalizerConfig] = None):
        self.config = config or TextNormalizerConfig()
        logger.debug("TextNormalizer initialized with config: %s", self.config)
 
    def normalize(self, text: str) -> NormalizationResult:
        """
        Apply full normalization pipeline to text.
 
        Args:
            text: Raw text from PDF extraction or other source
 
        Returns:
            NormalizationResult with normalized text and change log
        """
        if not text:
            return NormalizationResult(original_text=text, normalized_text=text)
 
        current = text
        changes: List[str] = []
 
        if self.config.unicode_form:
            normalized = unicodedata.normalize(self.config.unicode_form, current)
            if normalized != current:
                changes.append(f"unicode_{self.config.unicode_form}")
            current = normalized
 
        if self.config.fix_ligatures:
            before = current
            for ligature, replacement in LIGATURE_MAP.items():
                current = current.replace(ligature, replacement)
            if current != before:
                changes.append("ligatures")
 
        if self.config.remove_soft_hyphens:
            before = current
            current = current.replace("\u00ad", "")  # soft hyphen
            if current != before:
                changes.append("soft_hyphens")
 
        if self.config.normalize_quotes:
            before = current
            current = current.replace("\u201c", '"').replace("\u201d", '"')
            current = current.replace("\u2018", "'").replace("\u2019", "'")
            if current != before:
                changes.append("quotes")
 
        if self.config.normalize_dashes:
            before = current
            current = current.replace("\u2013", "-").replace("\u2014", "--")
            if current != before:
                changes.append("dashes")
 
        if self.config.strip_control_chars:
            before = current
            current = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", current)
            if current != before:
                changes.append("control_chars")
 
        if self.config.collapse_whitespace:
            before = current
            current = re.sub(r"[ \t]+", " ", current)
            current = re.sub(r"\n{3,}", "\n\n", current)
            current = current.strip()
            if current != before:
                changes.append("whitespace")
 
        return NormalizationResult(
            original_text=text,
            normalized_text=current,
            changes_made=changes,
        )
 
    def normalize_text(self, text: str) -> str:
        """Convenience method - returns normalized string directly."""
        return self.normalize(text).normalized_text
 
 
@lru_cache(maxsize=1)
def get_text_normalizer(
    unicode_form: str = "NFC",
    fix_ligatures: bool = True,
) -> TextNormalizer:
    """
    Factory function - returns cached TextNormalizer instance.
 
    Args:
        unicode_form: Unicode normalization form
        fix_ligatures: Whether to replace PDF ligature characters
 
    Returns:
        Cached TextNormalizer instance
    """
    config = TextNormalizerConfig(
        unicode_form=unicode_form,
        fix_ligatures=fix_ligatures,
    )
    return TextNormalizer(config)
