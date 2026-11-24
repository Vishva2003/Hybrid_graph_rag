import pdfplumber
from pathlib import Path
from typing import List, Dict
from llama_index.core.schema import Document

class DocumentLoader:
    @staticmethod
    def load_pdf(file_name) -> List[Documents]:
        file_path = Path(file_name)

        if not file_path.exists():
            raise FileNotFoundError(f'The file {file_name} does not exist.')

        documents = []

        with pdfplumber.open(file_path) as f:
            for page_no, page in enumerate(f.pages, start=1):
                page_text = page.extract_text()
                if not page_text or not page_text.strip():
                    continue
                
                doc_id = f"{file_path.name}::p{page_no}"

                documents.append(
                    Document(
                        doc_id = doc_id,
                        text=page_text.strip(),
                        metadata={
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "page": page_no,
                            "total_pages": len(f.pages),
                            "source": str(file_path),   # document identity
                            "loader": "pdfplumber",     # parser identity
                        },
                    )
                )
        
            if not documents:
                raise ValueError(
                    f"No extractable text found in {file_path.name}. "
                )

        return documents


if __name__ == "__main__":
    
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    file_path = PROJECT_ROOT / "data" / "raw" / "The+48+Laws+Of+Power.pdf"

    loader = DocumentLoader()
    docs = loader.load_pdf(file_path)

    print(f"Loaded {len(docs)} pages from {file_path.name}")
    print(f"First page metadata: {docs[0].metadata}")
    print("--- preview (first 500 chars of page 1) ---")
    print(docs[0].text[:500])