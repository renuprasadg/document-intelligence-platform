from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import logging
import re

logger = logging.getLogger(__name__)


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class CleanConfig:
    header_scan_lines: int = 10
    footer_scan_lines: int = 3
    # Keep this list small and general; project-specific patterns can be added later.
    header_footer_exact: tuple[str, ...] = (
        "Financial Conduct Authority",
        "FG22/5",
    )
    header_footer_prefixes: tuple[str, ...] = (
        "FG22/5 Final non-Handbook Guidance",
        "Chapter",
    )


# ----------------------------
# Helpers
# ----------------------------

def is_header_or_footer(line: str, cfg: CleanConfig) -> bool:
    s = line.strip()
    if not s:
        return False

    # page numbers like "1", "12", "103"
    if s.isdigit() and len(s) <= 3:
        return True

    if s in cfg.header_footer_exact:
        return True

    s_lower = s.lower()
    for pref in cfg.header_footer_prefixes:
        # allow case-insensitive match for "chapter"
        if pref.lower() == "chapter":
            if s_lower.startswith("chapter"):
                return True
        else:
            if s.startswith(pref):
                return True

    return False


# ----------------------------
# Core cleaning steps
# ----------------------------

def clean_page_text(text: str) -> str:
    """Normalize line endings/tabs and collapse repeated blank lines."""
    logger.debug("Cleaning page text: %d chars in", len(text))

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    # per-line trim (keep left spacing; PDFs sometimes indent intentionally)
    lines = [ln.rstrip() for ln in cleaned.split("\n")]
    cleaned = "\n".join(lines)

    # collapse multiple blank lines to max 1 blank line
    out_lines: list[str] = []
    prev_blank = False
    for line in cleaned.split("\n"):
        is_blank = (line.strip() == "")
        if is_blank and prev_blank:
            continue
        out_lines.append(line)
        prev_blank = is_blank

    cleaned = "\n".join(out_lines)

    logger.debug("Cleaning page text: %d chars out", len(cleaned))
    return cleaned


def remove_headers_footers(text: str, cfg: CleanConfig) -> str:
    """Remove likely header/footer lines from top/bottom of each page."""
    lines = text.split("\n")
    new_lines: list[str] = []

    for i, line in enumerate(lines):
        if i < cfg.header_scan_lines and is_header_or_footer(line, cfg):
            continue
        if i >= len(lines) - cfg.footer_scan_lines and is_header_or_footer(line, cfg):
            continue
        new_lines.append(line)

    return "\n".join(new_lines)


def join_wrapped_lines(text: str) -> str:
    """
    Join PDF-wrapped lines into paragraphs while preserving:
    - blank lines (paragraph breaks)
    - bullets
    - headings/section numbers
    - lines ending with sentence punctuation
    Also fixes simple hyphenation at line breaks: 'informa-\\ntion' -> 'information'
    """
    punct_end = (".", "!", "?", ";", ":", "”", '"', "’", "'")
    bullet_prefixes = ("•", "-", "*", "–", "—")
    section_re = re.compile(r"^\d+(\.\d+)*$")

    lines = text.split("\n")
    out: list[str] = []
    i = 0

    def is_bullet(line: str) -> bool:
        s = line.lstrip()
        return s.startswith(bullet_prefixes)

    def is_section_marker(line: str) -> bool:
        s = line.strip()
        return bool(section_re.match(s))

    while i < len(lines):
        cur = lines[i]
        cur_s = cur.strip()

        if cur_s == "":
            out.append("")
            i += 1
            continue

        if i == len(lines) - 1:
            out.append(cur.rstrip())
            break

        nxt = lines[i + 1]
        nxt_s = nxt.strip()

        if nxt_s == "":
            out.append(cur.rstrip())
            i += 1
            continue

        if is_bullet(cur) or is_bullet(nxt):
            out.append(cur.rstrip())
            i += 1
            continue

        if is_section_marker(cur) or is_section_marker(nxt):
            out.append(cur.rstrip())
            i += 1
            continue

        if len(cur_s) <= 40 and cur_s.istitle():
            out.append(cur.rstrip())
            i += 1
            continue

        if cur.rstrip().endswith(punct_end):
            out.append(cur.rstrip())
            i += 1
            continue

        # hyphenation: "informa-" + "tion" => "information"
        if cur.rstrip().endswith("-") and nxt_s and nxt_s[0].islower():
            merged = cur.rstrip()[:-1] + nxt.lstrip()
            out.append(merged)
            i += 2
            continue

        # join if next line continues a sentence
        if nxt_s and nxt_s[0].islower():
            merged = cur.rstrip() + " " + nxt.lstrip()
            out.append(merged)
            i += 2
            continue

        out.append(cur.rstrip())
        i += 1

    return "\n".join(out)


# ----------------------------
# PDF extraction + pipeline wrapper
# ----------------------------

def load_pdf_text_by_page(pdf_path: str | Path) -> list[str]:
    path = Path(pdf_path)
    logger.info("Loading PDF: %s", path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path.resolve()}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file but got: {path.name}")

    if path.stat().st_size == 0:
        raise ValueError(f"PDF is empty: {path.resolve()}")

    pages: list[str] = []  # IMPORTANT: define before try (fixes your finally bug)
    pdf = None

    try:
        pdf = fitz.open(path)
        logger.info("PDF opened successfully with %d pages", len(pdf))

        for i in range(len(pdf)):
            text = pdf[i].get_text()
            logger.debug("Extracting page %d (%d chars)", i + 1, len(text))
            pages.append(text)

        return pages

    finally:
        if pdf is not None:
            pdf.close()
        logger.info("Completed extraction: %d pages", len(pages))


def clean_pdf_to_pages(pdf_path: str | Path, cfg: CleanConfig | None = None) -> list[str]:
    """Full pipeline: extract -> normalize -> header/footer -> wrap-join."""
    if cfg is None:
        cfg = CleanConfig()

    raw_pages = load_pdf_text_by_page(pdf_path)
    cleaned_pages: list[str] = []

    for raw in raw_pages:
        safe = clean_page_text(raw)
        struct = remove_headers_footers(safe, cfg)
        final = join_wrapped_lines(struct)
        cleaned_pages.append(final)

    return cleaned_pages


def save_cleaned_pages_as_text(pages: Iterable[str], output_path: str | Path) -> Path:
    """
    Save cleaned pages into a single text file with clear page separators.
    Good for Week-1; later you can save JSON with metadata.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, page in enumerate(pages, start=1):
            f.write(f"\n\n===== PAGE {idx} =====\n\n")
            f.write(page.strip())
            f.write("\n")

    logger.info("Saved cleaned output to: %s", out_path.resolve())
    return out_path
