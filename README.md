hybrid-graphrag/                          # Your project root
│
├── data/
│   ├── raw/                              # Put your original PDFs, docs, txt here
│   ├── processed/                        # Cleaned & chunked data (auto-generated)
│   └── evaluation/                       # Golden questions + RAGAS results
│
├── notebooks/                            # Your learning & experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_chunking_and_embedding.ipynb
│   ├── 03_graph_building.ipynb
│   └── 04_ragas_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py                         # All settings in one place (very important)
│   │
│   ├── ingestion/
│   │   ├── loader.py                     # Load PDFs, TXT, Markdown...
│   │   └── chunker.py                    # Chunking logic
│   │
│   ├── indexing/
│   │   ├── embeddings.py                 # Dense embeddings
│   │   ├── bm25_index.py                 # BM25 sparse index
│   │   └── vector_store.py               # Qdrant / Chroma
│   │
│   ├── graph/
│   │   ├── extractor.py                  # Extract entities & relationships with LLM
│   │   ├── graph_store.py                # Neo4j or NetworkX
│   │   └── graph_rag.py                  # Local + Global search
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py           # BM25 + Vector + RRF fusion
│   │   ├── graph_retriever.py
│   │   └── reranker.py                   # Optional but recommended
│   │
│   ├── generation/
│   │   ├── prompts.py                    # All your prompts
│   │   └── generator.py                  # Final answer generation
│   │
│   ├── agents/
│   │   └── agent.py                      # LangGraph agent (final layer)
│   │
│   ├── evaluation/
│   │   └── ragas_eval.py                 # RAGAS evaluation
│   │
│   └── pipeline.py                       # Main orchestration (ingest → index → query)
│
├── scripts/                              # Easy-to-run commands
│   ├── 01_ingest.py
│   ├── 02_build_indexes.py               # Vector + BM25 + Graph
│   ├── 03_query.py
│   ├── 04_evaluate.py
│   └── 05_run_agent.py
│
├── docker-compose.yml                    # Qdrant + Neo4j (one command to start)
├── .env.example
├── requirements.txt                      # or pyproject.toml
├── README.md
└── main.py                               # Optional simple entry point
