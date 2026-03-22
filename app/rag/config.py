import os
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

# App paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_DIR = os.path.join(BASE_DIR, ".docs")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# ChromaDB settings
COLLECTION_NAME = "knowledge_base"

# Whisper specific
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large

# RAG specific  
EMBEDDING_MODEL = "models/gemini-embedding-001"  
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2026 Upgrades
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # For reranking
LLAMA_PARSE_KEY = os.getenv("LLMA_PRASE_KEY") # PDF Vision Parsing
USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() == "true"
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
USE_NEO4J = os.getenv("USE_NEO4J", "false").lower() == "true"  # Opt-in for now

# Rate Limiting & Retry Configuration (Optimized for Free Tier)
EMBEDDING_BATCH_SIZE = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "1")))  # Process one by one to stay within free-tier RPM
RETRY_MAX_ATTEMPTS = 5     # Max retry attempts for rate limits
RETRY_INITIAL_DELAY = 1    # Start with 1s delay
RETRY_MAX_DELAY = 10       # Max wait 10s
BATCH_DELAY = 1.0          # Mandatory 1s delay (faster)

# Logging Configuration
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

setup_logging()
