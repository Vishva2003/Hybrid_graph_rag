from typing import List
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """
    Handles embedding of TextNodes using local BAAI/bge-small-en-v1.5 model.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name

        self.embed_model = HuggingFaceEmbedding(
            model_name=self.model_name,
            trust_remote_code=True,
            # Optional: force CPU if you don't have GPU
            # device="cpu"
        )

        logger.info(f"Initialized Embedder with model: {self.model_name}")

    def embed_nodes(self, nodes: List[TextNode]) -> List[TextNode]:
        """
        Generate embeddings for a list of TextNodes and attach them.
        """
        if not nodes:
            logger.warning("No nodes provided for embedding.")
            return []

        logger.info(f"Generating embeddings for {len(nodes)} nodes using {self.model_name}...")

        texts = [node.get_content(metadata_mode="none") for node in nodes]

        # Generate embeddings in batch
        embeddings = self.embed_model.get_text_embedding_batch(texts, show_progress=True)

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        logger.info(f"Successfully embedded {len(nodes)} nodes.")
        return nodes

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.
        """
        return self.embed_model.get_query_embedding(query)


if __name__ == "__main__":
    embedder = Embedder()

    nodes = [
        TextNode(
            id_="The+48+Laws+Of+Power.pdf::p1::c0",
            text="LAW 1: Never outshine the master.",
            ref_doc_id="The+48+Laws+Of+Power.pdf::p1",
            metadata={
                "file_name": "The+48+Laws+Of+Power.pdf",
                "file_path": "C:\\Users\\harsa\\...\\data\\raw\\The+48+Laws+Of+Power.pdf",
                "page": 1,
                "total_pages": 476,
                "source": "C:\\Users\\harsa\\...\\data\\raw\\The+48+Laws+Of+Power.pdf",
                "loader": "pdfplumber",
                "chunk_id": 0,
                "global_chunk_id": 0,
                "char_length": 37,
            },
        ),
        TextNode(
            id_="The+48+Laws+Of+Power.pdf::p2::c0",
            text="LAW 2: Never put too much trust in friends, learn how to use enemies.",
            ref_doc_id="The+48+Laws+Of+Power.pdf::p2",
            metadata={
                "file_name": "The+48+Laws+Of+Power.pdf",
                "file_path": "C:\\Users\\harsa\\...\\data\\raw\\The+48+Laws+Of+Power.pdf",
                "page": 2,
                "total_pages": 476,
                "source": "C:\\Users\\harsa\\...\\data\\raw\\The+48+Laws+Of+Power.pdf",
                "loader": "pdfplumber",
                "chunk_id": 0,
                "global_chunk_id": 1,
                "char_length": 69,
            },
        ),
    ]
    doc_vectors = embedder.embed_nodes(nodes)
    print(f"Document vectors: {len(doc_vectors)} × {len(doc_vectors)} dims")

    query_vector = embedder.embed_query("How to deal with friends?")
    print(f"Query vector dims: {len(query_vector)}")