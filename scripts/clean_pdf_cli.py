"""
CLI: Extract and clean a PDF file (Enterprise Edition)
 
Usage:
  python -m scripts.clean_pdf_cli input.pdf --output-dir ./cleaned
  python -m scripts.clean_pdf_cli input.pdf --verbose
 
Outputs:
  <output-dir>/<filename>.txt  - Cleaned text
  <output-dir>/<filename>_report.json  - Quality report
"""
import argparse
import json
import sys
from pathlib import Path
 
from knowledge_engine.rag.cleaning.doc_cleaner import get_document_cleaner
from knowledge_engine.core.logging_config import get_logger, setup_logging
 
logger = get_logger(__name__)
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and clean a PDF for GuardianRAG ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument(
        "--output-dir", "-o",
        default="./cleaned",
        help="Output directory for cleaned files (default: ./cleaned)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--fail-on-low-quality",
        action="store_true",
        help="Exit with code 1 if document quality is LOW or REJECT",
    )
    return parser.parse_args()
 
 
def main() -> int:
    """
    Main entry point.
 
    Returns:
        Exit code (0 = success, 1 = failure)
    """
    args = parse_args()
    setup_logging("DEBUG" if args.verbose else "INFO")
 
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return 1
 
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    logger.info("Cleaning PDF: %s", pdf_path)
 
    cleaner = get_document_cleaner()
    result = cleaner.clean(str(pdf_path))
 
    if not result.succeeded:
        logger.error("Pipeline failed: %s", result.pipeline_error)
        return 1
 
    # Write cleaned text
    txt_path = output_dir / (pdf_path.stem + ".txt")
    txt_path.write_text(result.clean_text, encoding="utf-8")
    logger.info("Cleaned text written to: %s", txt_path)
 
    # Write quality report
    report_path = output_dir / (pdf_path.stem + "_report.json")
    if result.quality:
        report_data = {
            "source": str(pdf_path),
            "quality_level": result.quality.quality_level.value,
            "overall_score": result.quality.overall_score,
            "word_count": result.quality.word_count,
            "passed": result.quality.passed,
            "warnings": result.quality.warnings,
            "failed_checks": [
                {"name": c.name, "value": c.value, "threshold": c.threshold}
                for c in result.quality.failed_checks
            ],
        }
        report_path.write_text(
            json.dumps(report_data, indent=2), encoding="utf-8"
        )
        logger.info("Quality report written to: %s", report_path)
 
    # Print summary
    print(f"\n=== CLEAN RESULT ===")
    print(f"Source:  {pdf_path}")
    print(f"Output:  {txt_path}")
    print(f"Quality: {result.summary()}")
 
    if args.fail_on_low_quality and result.quality:
        from knowledge_engine.rag.cleaning.document_quality_validator import QualityLevel
        if result.quality.quality_level in (QualityLevel.LOW, QualityLevel.REJECT):
            logger.warning("Document quality below threshold - failing as requested")
            return 1
 
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
