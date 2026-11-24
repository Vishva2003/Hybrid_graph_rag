import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import Chunker

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    loader = DocumentLoader()
    chunker = Chunker(chunk_size=700, chunk_overlap=100)

    file_path = PROJECT_ROOT / "data" / "raw" / "The+48+Laws+Of+Power.pdf"

    if not file_path.exists():
        raise SystemExit(f"PDF not found at {file_path}")

    documents = loader.load_pdf(file_path)
    chunks = chunker.chunk_documents(documents)

    print(f"Loaded {len(documents)} pages → {len(chunks)} chunks")
    print(f"\nFirst chunk ({len(chunks[0].text)} chars):")
    print(chunks[0].text[:500])
    print(f"\nFirst chunk metadata: {chunks[0].metadata}")
    print(f"First chunk id_: {chunks[0].id_}")
    print(f"First chunk ref_doc_id: {chunks[0].ref_doc_id}")