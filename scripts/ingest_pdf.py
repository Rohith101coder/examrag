"""CLI entry point for ingesting local PDF documents into ExamRAG."""

import argparse
import sys
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import ExamRAGException
from app.core.logging import setup_logging
from app.services.ingestion_service import IngestionService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ExamRAG PDF Ingestion CLI — Index course PDFs into Qdrant Cloud."
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF document to ingest"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Optional override for target Qdrant collection name"
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.collection:
        settings.QDRANT_COLLECTION_NAME = args.collection

    logger = setup_logging(settings.LOG_LEVEL)

    file_path = Path(args.pdf_path).resolve()
    if not file_path.exists():
        print(f"Error: Specified PDF file does not exist: {file_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("ExamRAG Document Ingestion CLI")
    print(f"Target file: {file_path.name}")
    print(f"Collection:  {settings.QDRANT_COLLECTION_NAME}")
    print(f"Embedding:   {settings.EMBEDDING_MODEL}")
    print("=" * 60)

    try:
        service = IngestionService(settings=settings)
        result = service.ingest_pdf_file(file_path)

        print("\nIngestion Summary:")
        print(f"  Status:       {result.status.upper()}")
        print(f"  Document ID:  {result.document_id}")
        print(f"  Filename:     {result.filename}")
        print(f"  Page Count:   {result.page_count}")
        print(f"  Chunk Count:  {result.chunk_count}")
        print(f"  Content Hash: {result.content_hash}")
        print(f"  Duration:     {result.duration_seconds:.2f} seconds")
        print("=" * 60)
        print("Success: Document is indexed and ready for retrieval.")

    except ExamRAGException as e:
        print(f"\n[Ingestion Error] {e.message}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[Unexpected Error] {type(e).__name__}: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

