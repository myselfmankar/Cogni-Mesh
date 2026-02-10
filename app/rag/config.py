import os
import logging
from dotenv import load_dotenv

load_dotenv()

# App paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_DIR = os.path.join(BASE_DIR, ".docs")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# ChromaDB settings
COLLECTION_NAME = "knowledge_base"

# Whisper specific
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large

# RAG specific  
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI Embedding (2026)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2026 Upgrades
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # For reranking
USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() == "true"
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
USE_NEO4J = os.getenv("USE_NEO4J", "false").lower() == "true"  # Opt-in for now

# Rate Limiting & Retry Configuration
EMBEDDING_BATCH_SIZE = 10  # Process embeddings in smaller batches
RETRY_MAX_ATTEMPTS = 5     # Max retry attempts for rate limits
RETRY_INITIAL_DELAY = 1    # Initial delay in seconds
RETRY_MAX_DELAY = 60       # Max delay in seconds

# Logging Configuration
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

setup_logging()
