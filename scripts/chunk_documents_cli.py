"""
CLI: Chunk cleaned text files for GuardianRAG (Enterprise Edition)
 
Usage:
  python -m scripts.chunk_documents_cli ./cleaned --chunker sentence
  python -m scripts.chunk_documents_cli ./cleaned --chunker semantic --chunk-size 400
  python -m scripts.chunk_documents_cli ./cleaned --output chunks.jsonl
 
Outputs:
  JSONL file: each line = {"text": "...", "metadata": {...}}
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List
 
from knowledge_engine.rag.chunking.base_chunker import BaseChunker, ChunkConfig
from knowledge_engine.rag.chunking.chunk_metadata_builder import (
    DocumentMetadata, get_metadata_builder,
)
from knowledge_engine.rag.chunking.semantic_chunker import SemanticChunker
from knowledge_engine.rag.chunking.sentence_splitter import SentenceSplitter
from knowledge_engine.core.logging_config import get_logger, setup_logging
 
logger = get_logger(__name__)
 
 
def build_chunker(name: str, chunk_size: int, overlap: int) -> BaseChunker:
    """Create a chunker by name."""
    config = ChunkConfig(chunk_size=chunk_size, chunk_overlap=overlap)
    if name == "sentence":
        return SentenceSplitter()
    elif name == "semantic":
        return SemanticChunker()
    else:
        from knowledge_engine.rag.chunking.base_chunker import SlidingWindowChunker
        return SlidingWindowChunker()
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk cleaned text files for GuardianRAG embedding"
    )
    parser.add_argument("input_dir", help="Directory containing .txt files")
    parser.add_argument(
        "--chunker", "-c",
        choices=["sliding", "sentence", "semantic"],
        default="semantic",
        help="Chunking strategy (default: semantic)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512,
        help="Target chunk size in words (default: 512)",
    )
    parser.add_argument(
        "--overlap", type=int, default=50,
        help="Overlap in words between chunks (default: 50)",
    )
    parser.add_argument(
        "--output", "-o",
        default="chunks.jsonl",
        help="Output JSONL file (default: chunks.jsonl)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()
 
 
def main() -> int:
    args = parse_args()
    setup_logging("DEBUG" if args.verbose else "INFO")
 
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        return 1
 
    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s", input_dir)
        return 0
 
    logger.info("Found %d text files in %s", len(txt_files), input_dir)
 
    chunker = build_chunker(args.chunker, args.chunk_size, args.overlap)
    builder = get_metadata_builder()
    output_path = Path(args.output)
 
    total_chunks = 0
 
    with output_path.open("w", encoding="utf-8") as out_f:
        for txt_path in sorted(txt_files):
            text = txt_path.read_text(encoding="utf-8")
            if not text.strip():
                logger.warning("Skipping empty file: %s", txt_path)
                continue
 
            chunk_result = chunker.chunk(text, source=str(txt_path))
            doc_meta = DocumentMetadata.from_path(str(txt_path))
            doc_meta.total_chars = len(text)
 
            meta_result = builder.build(chunk_result, doc_meta)
 
            for enriched in meta_result.enriched_chunks:
                line = json.dumps({
                    "text": enriched.text,
                    "metadata": enriched.metadata,
                }, ensure_ascii=False)
                out_f.write(line + "\n")
                total_chunks += 1
 
            logger.info(
                "  %s -> %d chunks",
                txt_path.name, len(meta_result.enriched_chunks)
            )
 
    print(f"\n=== CHUNKING COMPLETE ===")
    print(f"Files processed: {len(txt_files)}")
    print(f"Total chunks:    {total_chunks}")
    print(f"Output:          {output_path}")
    print(f"Chunker:         {chunker.name}")
 
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
