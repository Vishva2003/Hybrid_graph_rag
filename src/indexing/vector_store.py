import logging
from pathlib import Path
from typing import List, Optional, Tuple

from llama_index.core.schema import NodeWithScore, TextNode
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384  # bge-small-en-v1.5 output dims


class VectorStore:
    """
    Qdrant dense vector store with FastEmbed integration.
    Mirrors BM25Indexer's API: index(), search_ids(), retrieve().
    """

    def __init__(
        self,
        collection_name: str = "hybrid_graphrag",
        path: Optional[str] = None,
        url: Optional[str] = None,
    ):
        self.collection_name = collection_name
        if path:
            self.client = QdrantClient(path=path)
        elif url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(":memory:")

        self._ensure_collection()
        self._nodes: List[TextNode] = []

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection %r", self.collection_name)

    # ---------- build ----------
    def index(self, nodes: List[TextNode]) -> None:
        """Upsert TextNodes. Uses node.id_ as the Qdrant point ID."""
        if not nodes:
            raise ValueError("No nodes provided to index")

        self._nodes = nodes

        # Qdrant requires point IDs to be int or UUID. Our IDs are strings
        # like "file.pdf::p1::c0", so we use UUID5 derived from the string
        # to get deterministic, collision-resistant UUIDs.
        import uuid

        points = []
        for node in nodes:
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, node.id_))
            payload = {
                **node.metadata,
                "node_id": node.id_,          # preserve original string ID
                "ref_doc_id": node.ref_doc_id,
                "text": node.text,            # store text for retrieval
            }
            points.append(
                models.PointStruct(
                    id=point_uuid,
                    vector=models.Document(
                        text=node.text, model=MODEL_NAME
                    ),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info("Upserted %d points into %r", len(points), self.collection_name)

    # ---------- search: ids only (for RRF fusion) ----------
    def search_ids(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return (node_id, score) pairs, mirroring BM25Indexer.search_ids()."""
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(text=query, model=MODEL_NAME),
            limit=top_k,
        ).points

        return [
            (hit.payload.get("node_id", str(hit.id)), float(hit.score))
            for hit in hits
        ]

    # ---------- search: LlamaIndex-native ----------
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """Return NodeWithScore objects, mirroring BM25Indexer.retrieve()."""
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(text=query, model=MODEL_NAME),
            limit=top_k,
        ).points

        results = []
        for hit in hits:
            payload = hit.payload or {}
            node = TextNode(
                id_=payload.get("node_id", str(hit.id)),
                text=payload.get("text", ""),
                ref_doc_id=payload.get("ref_doc_id"),
                metadata={k: v for k, v in payload.items()
                        if k not in ("text", "node_id", "ref_doc_id")},
            )
            results.append(NodeWithScore(node=node, score=float(hit.score)))

        logger.debug("Vector retrieved %d nodes for %r", len(results), query[:50])
        return results

    def get_index_size(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

# unit testing 


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    nodes = [
        TextNode(
            id_="The+48+Laws+Of+Power.pdf::p1::c0",
            text="LAW 1: Never outshine the master.",
            ref_doc_id="The+48+Laws+Of+Power.pdf::p1",
            metadata={"file_name": "The+48+Laws+Of+Power.pdf", "page": 1},
        ),
        TextNode(
            id_="The+48+Laws+Of+Power.pdf::p2::c0",
            text="LAW 2: Never put too much trust in friends, learn how to use enemies.",
            ref_doc_id="The+48+Laws+Of+Power.pdf::p2",
            metadata={"file_name": "The+48+Laws+Of+Power.pdf", "page": 2},
        ),
        TextNode(
            id_="The+48+Laws+Of+Power.pdf::p3::c0",
            text="LAW 3: Conceal your intentions. Keep people off-balance.",
            ref_doc_id="The+48+Laws+Of+Power.pdf::p3",
            metadata={"file_name": "The+48+Laws+Of+Power.pdf", "page": 3},
        ),
    ]

    store = VectorStore()
    store.index(nodes)
    print(f"Index size: {store.get_index_size()} points\n")

    query = "How to use enemies?"
    results = store.retrieve(query, top_k=2)

    print(f"Query: {query}\n" + "-" * 60)
    for i, r in enumerate(results, 1):
        print(f"Rank {i} | Score: {r.score:.4f}")
        print(f"id_        : {r.node.id_}")
        print(f"ref_doc_id : {r.node.ref_doc_id}")
        print(f"Text       : {r.node.text}")
        print(f"Page       : {r.node.metadata.get('page')}")
        print("-" * 60)

    print("\nsearch_ids (for RRF fusion):")
    print(store.search_ids(query, top_k=2))