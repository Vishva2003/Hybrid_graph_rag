from typing import List, Dict, Any
from llama_index.core.schema import TextNode
from llama_index.llms.google_genai import GoogleGenAI
from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EXTRACTOR_MODEL, GOOGLE_API_KEY
import json
import logging
import re

logger = logging.getLogger(__name__)


class GraphExtractor:
    """
    Extracts entities and relationships using OpenRouter.
    """

    def __init__(
        self,
        model: str = EXTRACTOR_MODEL,          # You can change this
        temperature: float = 0.0
    ):
        self.llm = GoogleGenAI(
            model=model,
            api_key= GOOGLE_API_KEY,  
            temperature=temperature,
            max_tokens= 1024
        )

        self.system_prompt = """
        You are an expert knowledge graph extractor.
        Extract important entities and relationships from the given text.

        Return ONLY valid JSON in this exact format:

        {
            "entities": [
                {"name": "Entity Name", "type": "Person|Concept|Law|Organization|Location|Other"}
            ],
            "relationships": [
                {"source": "Entity A", "target": "Entity B", "relation": "short relationship description"}
            ]
        }
"""

    def _clean_json(self, text: str) -> str:
        """Clean markdown formatting from LLM response"""
        text = text.strip()
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        return text.strip()

    def extract_from_text(self, text: str) -> Dict[str, Any]:
        prompt = f"""{self.system_prompt}

Text:
{text}

Return only the JSON:"""

        try:
            response = self.llm.complete(prompt)
            cleaned = self._clean_json(response.text)
            data = json.loads(cleaned)
            return data
        except Exception as e:
            logger.warning(f"Failed to extract/parse: {e}")
            return {"entities": [], "relationships": []}

    def extract_from_nodes(self, nodes: List[TextNode]) -> List[Dict[str, Any]]:
        results = []

        for i, node in enumerate(nodes, 1):
            logger.info(f"Extracting node {i}/{len(nodes)}...")
            text = node.get_content(metadata_mode="none")
            extraction = self.extract_from_text(text)

            results.append({
                "node_id": node.node_id,
                "entities": extraction.get("entities", []),
                "relationships": extraction.get("relationships", []),
                "metadata": node.metadata
            })

        logger.info(f"Finished extracting from {len(nodes)} nodes")
        return results
    
    
#unit test

if __name__ == "__main__":
    import pickle
    from pathlib import Path
    extractor = GraphExtractor()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    NODES_PKL = PROJECT_ROOT / "data" / "processed" / "nodes_with_embeddings.pkl"

    if not NODES_PKL.exists():
        raise SystemExit(f"Nodes pickle not found: {NODES_PKL}")

    with open(NODES_PKL, "rb") as f:
        nodes = pickle.load(f)
    logger.info("Loaded %d nodes from %s", len(nodes), NODES_PKL)

    # Extract from your nodes
    graph_data = extractor.extract_from_nodes(nodes[:5])   # Start with few nodes for testing

    for item in graph_data:
        print("Entities:", item["entities"])
        print("Relationships:", item["relationships"])
        print("-" * 50)