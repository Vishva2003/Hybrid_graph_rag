from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
from llama_index.core.schema import TextNode, NodeWithScore
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class GraphRAG:
    """
    Simple but effective GraphRAG implementation using NetworkX.
    Supports:
    - Building knowledge graph from extractions
    - Entity-based retrieval
    - 1-hop and 2-hop neighborhood retrieval
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.entity_to_nodes: Dict[str, List[str]] = defaultdict(list)  # entity -> list of node_ids
        self.node_id_to_node: Dict[str, TextNode] = {}

    def build_from_extractions(self, extractions: List[Dict[str, Any]], nodes: List[TextNode]):
        """
        Build the knowledge graph from extractor results + original nodes.
        """
        # Map node_id → TextNode for later retrieval
        self.node_id_to_node = {node.node_id: node for node in nodes}

        for item in extractions:
            node_id = item["node_id"]
            entities = item.get("entities", [])
            relationships = item.get("relationships", [])

            # Add entities as nodes in the graph
            for entity in entities:
                name = entity["name"].strip()
                etype = entity.get("type", "Other")

                if not name:
                    continue

                # Add entity node
                if not self.graph.has_node(name):
                    self.graph.add_node(name, type=etype)

                # Link entity → original text chunk
                self.entity_to_nodes[name].append(node_id)

            # Add relationships as edges
            for rel in relationships:
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                relation = rel.get("relation", "related_to")

                if source and target:
                    self.graph.add_edge(source, target, relation=relation)

        logger.info(f"Graph built → Nodes: {self.graph.number_of_nodes()} | Edges: {self.graph.number_of_edges()}")

    def get_related_entities(self, entity: str, max_hops: int = 1) -> List[str]:
        """
        Get related entities within max_hops.
        """
        if entity not in self.graph:
            return []

        if max_hops == 1:
            return list(self.graph.neighbors(entity))
        
        # 2-hop
        related = set()
        for neighbor in self.graph.neighbors(entity):
            related.add(neighbor)
            related.update(self.graph.neighbors(neighbor))
        related.discard(entity)
        return list(related)

    def retrieve(
        self,
        query_entities: List[str],
        top_k: int = 5,
        max_hops: int = 1
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant text chunks based on entities found in the query.
        """
        if not query_entities:
            return []

        # Collect relevant node_ids
        candidate_node_ids = set()
        entity_scores = defaultdict(float)

        for entity in query_entities:
            # Direct matches
            if entity in self.entity_to_nodes:
                for nid in self.entity_to_nodes[entity]:
                    candidate_node_ids.add(nid)
                    entity_scores[nid] += 2.0  # Higher score for direct match

            # Related entities (1-hop or 2-hop)
            related = self.get_related_entities(entity, max_hops=max_hops)
            for rel_entity in related:
                if rel_entity in self.entity_to_nodes:
                    for nid in self.entity_to_nodes[rel_entity]:
                        candidate_node_ids.add(nid)
                        entity_scores[nid] += 1.0

        # Build ranked results
        results = []
        for node_id in candidate_node_ids:
            node = self.node_id_to_node.get(node_id)
            if node:
                score = entity_scores[node_id]
                results.append(NodeWithScore(node=node, score=score))

        # Sort by score
        results = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

        logger.info(f"GraphRAG retrieved {len(results)} nodes")
        return results

    def get_graph_stats(self) -> Dict[str, Any]:
        return {
            "num_entities": self.graph.number_of_nodes(),
            "num_relationships": self.graph.number_of_edges(),
            "num_linked_chunks": len(self.node_id_to_node)
        }

    def save(self, path: str):
        """Save graph to disk"""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "graph": self.graph,
                "entity_to_nodes": dict(self.entity_to_nodes),
                "node_id_to_node": self.node_id_to_node
            }, f)
        logger.info(f"Graph saved to {path}")

    def load(self, path: str):
        """Load graph from disk"""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.graph = data["graph"]
            self.entity_to_nodes = defaultdict(list, data["entity_to_nodes"])
            self.node_id_to_node = data["node_id_to_node"]
        logger.info(f"Graph loaded from {path}")