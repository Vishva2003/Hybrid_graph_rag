import logging
import uuid

from pathlib import Path
from typing import List
from llama_index.core.schema import Document, TextNode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_OVERLAP,CHUNK_SIZE

logger = logging.getLogger(__name__)
class Chunker:
    """Chunk documents using RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        nodes = []
        global_idx = 0

        for doc in documents:
            chunks = self.text_splitter.split_text(doc.text)
            page = doc.metadata.get("page")

            for local_idx, chunk in enumerate(chunks):

                node_id = (
                    f"{doc.metadata.get('file_name', 'doc')}::p{page}::c{local_idx}"
                )
                node = TextNode(
                    id_=node_id,
                    text=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": local_idx,
                        "global_chunk_id": global_idx,
                        "char_length": len(chunk),
                    },
                    ref_doc_id=doc.doc_id,
                )
                nodes.append(node)
                global_idx += 1

        logger.info("Created %d chunks from %d documents", len(nodes), len(documents))
        return nodes