import argparse
import logging
from pathlib import Path

from knowledge_engine.rag.cleaning.doc_cleaner import (
    CleanConfig,
    clean_pdf_to_pages,
    save_cleaned_pages_as_text,
)

def main():
    parser = argparse.ArgumentParser(description="Clean a PDF and export cleaned text")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument(
        "--out",
        required=False,
        help="Output .txt path (default: data/processed/<pdfname>.cleaned.txt)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    pdf_path = Path(args.pdf)
    out_path = Path(args.out) if args.out else Path("data/processed") / f"{pdf_path.stem}.cleaned.txt"

    cfg = CleanConfig()
    pages = clean_pdf_to_pages(pdf_path, cfg=cfg)
    save_cleaned_pages_as_text(pages, out_path)

if __name__ == "__main__":
    main()

