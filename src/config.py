import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"



#Openrouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")


# Model settings
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32

#Extractor
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Reranking settings
CROSS_ENCODER_MODEL = "cross-encoder/qnli-distilroberta-base"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RESULTS = 20

# Create directories if they don't exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    '.pdf': 'pdf',
    '.txt': 'text',
    '.doc': 'word',
    '.docx': 'word'
}

models = {
        "Chatgpt-5": "openai/gpt-5-chat",
        "Grok-4-Fast": "x-ai/grok-4-fast",
        "Claude-Sonnet-4.5": "anthropic/claude-sonnet-4.5",
        "Deepseek-v3": "deepseek/deepseek-chat-v3.1",
        "Gemini-2.5-Flash-lite": "google/gemini-2.5-flash-lite",
        "Qwen3.5-flash": "qwen/qwen3.5-flash-02-23",
        "MiniMax-m1": "minimax/minimax-m1",
        "Ministral-3": "mistralai/ministral-3b-2512"
    }