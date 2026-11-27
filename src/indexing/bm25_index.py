from typing import List, Optional
from rank_bm25 import BM25Okapi
from llama_index.core.schema import TextNode, NodeWithScore
import logging
import re


logger = logging.getLogger(__name__)


class BM25Indexer:
    """
    Simple and effective BM25 sparse retriever.
    Works with LlamaIndex TextNodes.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.nodes: List[TextNode] = []
        self.corpus_tokens: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple but effective tokenization"""
        
        text = text.lower()
        # Keep only alphanumeric and basic punctuation
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        return tokens

    def index(self, nodes: List[TextNode]):
        """
        Build the BM25 index from a list of TextNodes.
        """
        if not nodes:
            raise ValueError("No nodes provided to index")

        self.nodes = nodes
        self.corpus_tokens = [self._tokenize(node.get_content(metadata_mode="none")) for node in nodes]

        self.bm25 = BM25Okapi(self.corpus_tokens)

        logger.info(f"BM25 index built successfully with {len(nodes)} documents")

    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieve top_k most relevant nodes using BM25.
        """
        if self.bm25 is None:
            raise ValueError("BM25 index is not built yet. Call .index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:  # Only return nodes with positive score
                results.append(
                    NodeWithScore(node=self.nodes[idx], score=score)
                )

        logger.info(f"BM25 retrieved {len(results)} nodes for query: '{query[:50]}...'")
        return results

    def get_index_size(self) -> int:
        return len(self.nodes)

# unit testing 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    nodes = [
        TextNode(
            id_="The+48+Laws+Of+Power.pdf::p1::c0",
            text="LAW 1: Never outshine the master.",
            ref_doc_id="The+48+Laws+Of+Power.pdf::p1",
            metadata={"file_name": "The+48+Laws+Of+Power.pdf", "page": 1, "chunk_id": 0, "global_chunk_id": 0, "char_length": 37},
        ),
        TextNode(
            id_="The+48+Laws+Of+Power.pdf::p2::c0",
            text="LAW 2: Never put too much trust in friends, learn how to use enemies.",
            ref_doc_id="The+48+Laws+Of+Power.pdf::p2",
            metadata={"file_name": "The+48+Laws+Of+Power.pdf", "page": 2, "chunk_id": 0, "global_chunk_id": 1, "char_length": 69},
        ),
        TextNode(
            id_="The+48+Laws+Of+Power.pdf::p3::c0",
            text="LAW 3: Conceal your intentions. Keep people off-balance.",
            ref_doc_id="The+48+Laws+Of+Power.pdf::p3",
            metadata={"file_name": "The+48+Laws+Of+Power.pdf", "page": 3, "chunk_id": 0, "global_chunk_id": 2, "char_length": 55},
        ),
    ]

    # 1. Create and build the index
    idx = BM25Indexer()
    idx.index(nodes)

    print(f"Index size: {idx.get_index_size()} documents\n")

    # 2. Test retrieval
    query = "conceal your intentions"
    results = idx.retrieve(query, top_k=3)

    print(f"Query: {query}\n")
    print("-" * 60)

    for i, r in enumerate(results, 1):
        print(f"Rank {i} | Score: {r.score:.4f}")
        print(f"id_        : {r.node.id_}")
        print(f"ref_doc_id : {r.node.ref_doc_id}")
        print(f"Text       : {r.node.text}")
        print(f"Page       : {r.node.metadata.get('page')}")
        print("-" * 60)