"""
Document quality validation for GuardianRAG pipeline (Enterprise Edition)
 
Scores documents across multiple quality dimensions and emits
a QualityReport dataclass with pass/fail status and per-check details.
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional
 
from knowledge_engine.core.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
class QualityLevel(str, Enum):
    """Document quality classification"""
    HIGH   = "high"    # Score >= 0.80
    MEDIUM = "medium"  # Score >= 0.60
    LOW    = "low"     # Score >= 0.40
    REJECT = "reject"  # Score < 0.40
 
 
@dataclass
class QualityCheck:
    """Result of a single quality check"""
    name: str
    passed: bool
    score: float                  # 0.0 - 1.0
    value: float                  # Actual measured value
    threshold: float              # Threshold that was checked
    message: str = ""
 
 
@dataclass
class QualityReport:
    """Full quality report for a document"""
    text_length: int
    word_count: int
    overall_score: float
    quality_level: QualityLevel
    checks: List[QualityCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
 
    @property
    def passed(self) -> bool:
        return self.quality_level != QualityLevel.REJECT
 
    @property
    def failed_checks(self) -> List[QualityCheck]:
        return [c for c in self.checks if not c.passed]
 
    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] Quality={self.quality_level.value} ({self.overall_score:.2f}) | "
            f"Words={self.word_count} | Failed checks: {len(self.failed_checks)}"
        )
 
 
@dataclass
class ValidatorConfig:
    """Thresholds for quality checks"""
    min_word_count: int = 50
    min_word_density: float = 0.3    # word chars / total chars
    min_avg_word_len: float = 2.5
    max_avg_word_len: float = 15.0
    max_punct_ratio: float = 0.20
    max_uppercase_ratio: float = 0.50
    max_non_ascii_ratio: float = 0.15
 
 
class DocumentQualityValidator:
    """
    Validates document quality before it enters the chunking pipeline.
 
    Usage:
        validator = get_quality_validator()
        report = validator.validate(cleaned_text)
        if not report.passed:
            logger.warning("Document quality too low: %s", report.summary())
    """
 
    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()
 
    def validate(self, text: str) -> QualityReport:
        """
        Run all quality checks on text.
 
        Args:
            text: Cleaned text to validate
 
        Returns:
            QualityReport with overall score and per-check results
        """
        words = text.split() if text else []
        word_count = len(words)
        text_length = len(text)
 
        checks: List[QualityCheck] = []
        warnings: List[str] = []
 
        # Check 1: Minimum word count
        checks.append(self._check_word_count(word_count))
 
        # Check 2: Word density
        checks.append(self._check_word_density(text, word_count))
 
        # Check 3: Average word length
        checks.append(self._check_avg_word_len(words))
 
        # Check 4: Punctuation ratio
        checks.append(self._check_punct_ratio(text, text_length))
 
        # Check 5: Uppercase ratio
        checks.append(self._check_uppercase_ratio(text))
 
        # Check 6: Non-ASCII ratio
        checks.append(self._check_non_ascii_ratio(text, text_length))
 
        # Overall score = weighted average
        if checks:
            overall_score = sum(c.score for c in checks) / len(checks)
        else:
            overall_score = 0.0
 
        if overall_score >= 0.80:
            level = QualityLevel.HIGH
        elif overall_score >= 0.60:
            level = QualityLevel.MEDIUM
            warnings.append("Medium quality document - review before production use")
        elif overall_score >= 0.40:
            level = QualityLevel.LOW
            warnings.append("Low quality document - may produce poor RAG results")
        else:
            level = QualityLevel.REJECT
            warnings.append("Document quality too low - consider excluding from index")
 
        return QualityReport(
            text_length=text_length,
            word_count=word_count,
            overall_score=round(overall_score, 4),
            quality_level=level,
            checks=checks,
            warnings=warnings,
        )
 
    def _check_word_count(self, word_count: int) -> QualityCheck:
        passed = word_count >= self.config.min_word_count
        score = min(1.0, word_count / max(self.config.min_word_count, 1))
        return QualityCheck("word_count", passed, min(score, 1.0), word_count,
            self.config.min_word_count,
            f"Word count {word_count} vs min {self.config.min_word_count}")
 
    def _check_word_density(self, text: str, word_count: int) -> QualityCheck:
        word_chars = sum(len(w) for w in text.split()) if text else 0
        density = word_chars / max(len(text), 1)
        passed = density >= self.config.min_word_density
        score = min(density / self.config.min_word_density, 1.0)
        return QualityCheck("word_density", passed, score, round(density, 3),
            self.config.min_word_density, f"Word density {density:.3f}")
 
    def _check_avg_word_len(self, words: list) -> QualityCheck:
        avg = sum(len(w) for w in words) / max(len(words), 1)
        passed = self.config.min_avg_word_len <= avg <= self.config.max_avg_word_len
        score = 1.0 if passed else 0.3
        return QualityCheck("avg_word_len", passed, score, round(avg, 2),
            self.config.min_avg_word_len, f"Avg word length {avg:.2f}")
 
    def _check_punct_ratio(self, text: str, text_length: int) -> QualityCheck:
        punct_count = len(re.findall(r"[^\w\s]", text))
        ratio = punct_count / max(text_length, 1)
        passed = ratio <= self.config.max_punct_ratio
        score = max(0.0, 1.0 - ratio / self.config.max_punct_ratio)
        return QualityCheck("punct_ratio", passed, score, round(ratio, 3),
            self.config.max_punct_ratio, f"Punct ratio {ratio:.3f}")
 
    def _check_uppercase_ratio(self, text: str) -> QualityCheck:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return QualityCheck("uppercase_ratio", True, 1.0, 0.0, self.config.max_uppercase_ratio)
        ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        passed = ratio <= self.config.max_uppercase_ratio
        score = max(0.0, 1.0 - ratio / self.config.max_uppercase_ratio)
        return QualityCheck("uppercase_ratio", passed, score, round(ratio, 3),
            self.config.max_uppercase_ratio, f"Uppercase ratio {ratio:.3f}")
 
    def _check_non_ascii_ratio(self, text: str, text_length: int) -> QualityCheck:
        non_ascii = sum(1 for c in text if ord(c) > 127)
        ratio = non_ascii / max(text_length, 1)
        passed = ratio <= self.config.max_non_ascii_ratio
        score = max(0.0, 1.0 - ratio / self.config.max_non_ascii_ratio)
        return QualityCheck("non_ascii_ratio", passed, score, round(ratio, 3),
            self.config.max_non_ascii_ratio, f"Non-ASCII ratio {ratio:.3f}")
 
 
@lru_cache(maxsize=1)
def get_quality_validator() -> DocumentQualityValidator:
    """Factory function - returns cached DocumentQualityValidator."""
    return DocumentQualityValidator()
