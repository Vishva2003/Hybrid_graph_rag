import sys
import logging
from pathlib import Path
import pickle

# ======================
# Fix import path
# ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import Chunker
from src.indexing.embeddings import Embedder
from src.indexing.bm25_index import BM25Indexer

# ======================
# Setup logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # --------------------------
    # 1. Paths
    # --------------------------
    data_dir = PROJECT_ROOT / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    indexes_dir = data_dir / "indexes"

    processed_dir.mkdir(parents=True, exist_ok=True)
    indexes_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = raw_dir / "The+48+Laws+Of+Power.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    # --------------------------
    # 2. Load & Chunk
    # --------------------------
    logger.info("Loading and chunking document...")
    loader = DocumentLoader()
    chunker = Chunker(chunk_size=700, chunk_overlap=100)

    documents = loader.load_pdf(pdf_path)
    nodes = chunker.chunk_documents(documents)

    logger.info(f"Created {len(nodes)} chunks")

    # --------------------------
    # 3. Create Embeddings
    # --------------------------
    logger.info("Generating embeddings with BAAI/bge-small-en-v1.5...")
    embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
    embedded_nodes = embedder.embed_nodes(nodes)

    # --------------------------
    # 4. Build BM25 Index
    # --------------------------
    logger.info("Building BM25 index...")
    bm25 = BM25Indexer()
    bm25.index(embedded_nodes)

    # --------------------------
    # 5. Save everything
    # --------------------------
    # Save nodes (with embeddings)
    nodes_path = processed_dir / "nodes_with_embeddings.pkl"
    with open(nodes_path, "wb") as f:
        pickle.dump(embedded_nodes, f)
    logger.info(f"Saved nodes → {nodes_path}")

    # Save BM25 index
    bm25_path = indexes_dir / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"Saved BM25 index → {bm25_path}")

    logger.info("=" * 60)
    logger.info("Indexing completed successfully!")
    logger.info(f"Total nodes      : {len(embedded_nodes)}")
    logger.info(f"Embedding dim    : {len(embedded_nodes[0].embedding)}")
    logger.info(f"BM25 documents   : {bm25.get_index_size()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()