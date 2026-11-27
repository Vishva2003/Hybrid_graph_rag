from typing import Optional
from pathlib import Path
import pickle
import logging
from src.graph.graph_rag import GraphRAG

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Handles saving and loading of the GraphRAG knowledge graph.
    """

    def __init__(self, storage_dir: str = "data/indexes"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.graph_path = self.storage_dir / "graph_rag.pkl"

    def save(self, graph_rag: GraphRAG) -> None:
        """
        Save the entire GraphRAG object to disk.
        """
        try:
            with open(self.graph_path, "wb") as f:
                pickle.dump(graph_rag, f)
            logger.info(f"GraphRAG saved successfully → {self.graph_path}")
        except Exception as e:
            logger.error(f"Failed to save GraphRAG: {e}")
            raise

    def load(self) -> Optional[GraphRAG]:
        """
        Load the GraphRAG object from disk.
        Returns None if the file does not exist.
        """
        if not self.graph_path.exists():
            logger.warning(f"No saved graph found at {self.graph_path}")
            return None

        try:
            with open(self.graph_path, "rb") as f:
                graph_rag = pickle.load(f)
            logger.info(f"GraphRAG loaded successfully from {self.graph_path}")
            return graph_rag
        except Exception as e:
            logger.error(f"Failed to load GraphRAG: {e}")
            raise

    def exists(self) -> bool:
        """Check if a saved graph already exists"""
        return self.graph_path.exists()

    def delete(self) -> None:
        """Delete the saved graph"""
        if self.graph_path.exists():
            self.graph_path.unlink()
            logger.info("Saved graph deleted.")