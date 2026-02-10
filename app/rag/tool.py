import logging
import chromadb
from typing import List, Optional
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.program.openai import OpenAIPydanticProgram
from . import config

# Setup Logging
logger = logging.getLogger(__name__)

# --- Structured Models ---
class Task(BaseModel):
    """
    Represents a single actionable task extracted from the context.
    """
    title: str = Field(..., description="Title of the task")
    description: str = Field(..., description="Detailed description")
    status: Optional[str] = Field(default="Pending", description="Status (Pending, In Progress, Done). Defaults to Pending if not specified.")

class RagResponse(BaseModel):
    """
    Structured response containing summary, insights, and actionable tasks derived from the knowledge base.
    """
    summary: str = Field(..., description="Comprehensive summary of the matched content")
    key_insights: List[str] = Field(
        ..., 
        description="List of 2-5 key insights derived from the content. Extract the most important findings."
    )
    tasks: List[Task] = Field(
        default_factory=list,
        description="ALL actionable tasks identified in the content. Extract every task mentioned, no matter how many. Can be 0 to 20+ tasks depending on content."
    )
    sources: List[str] = Field(..., description="List of source filenames used")

# --- RAG Tool Class ---
class RagAgentTool:
    def __init__(self):
        self.initialized = False
        self.query_fn = None
        self._initialize()

    def _initialize(self):
        try:
            """
            EMBEDDING MODEL OPTIONS:
            Currently using: OpenAI text-embedding-3-small 
            ALTERNATIVES (if hitting rate limits or want to avoid costs):
            
            1. OPENAI MODELS (Paid, Best Quality)
               - text-embedding-3-small (current) - Fast, cheap, good quality
               - text-embedding-3-large - Higher quality, more expensive
               - text-embedding-ada-002 - Legacy, still good
               
               Usage:
               embed_model = OpenAIEmbedding(model="text-embedding-3-large")
            
            2. HUGGINGFACE MODELS (Free, Local, No Rate Limits!)
               - BAAI/bge-small-en-v1.5 - Fast, good for general use
               - BAAI/bge-base-en-v1.5 - Better quality, slower
               - sentence-transformers/all-MiniLM-L6-v2 - Very fast, lightweight
               - thenlper/gte-large - High quality
               
               Usage:
               from llama_index.embeddings.huggingface import HuggingFaceEmbedding
               embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            
            3. OLLAMA (Free, Local, Privacy-Focused)
               - nomic-embed-text - Good quality, runs locally
               - mxbai-embed-large - High quality embeddings
               
               Requirements: Install Ollama first (ollama.ai)
               Usage:
               from llama_index.embeddings.ollama import OllamaEmbedding
               embed_model = OllamaEmbedding(model_name="nomic-embed-text")
            
            4. COHERE (Paid Alternative to OpenAI)
               - embed-english-v3.0 - Good quality, different pricing
               
               Usage:
               from llama_index.embeddings.cohere import CohereEmbedding
               embed_model = CohereEmbedding(api_key="your-key", model_name="embed-english-v3.0")
            
            5. GOOGLE VERTEX AI (Google Cloud)
               - textembedding-gecko - Google's embedding model
               Usage:
               from llama_index.embeddings.vertex import VertexTextEmbedding
               embed_model = VertexTextEmbedding()
            """
            
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            embed_model = OpenAIEmbedding(
                model=config.EMBEDDING_MODEL,
                max_retries=config.RETRY_MAX_ATTEMPTS
            )
            Settings.embed_model = embed_model
            
            Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

            logger.info(f"Connecting to ChromaDB at {config.CHROMA_DB_DIR}")
            db = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
            chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )
            
            self.retriever = index.as_retriever(similarity_top_k=5)
            self.initialized = True
            logger.info("RAG Agent Tool Initialized Successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent Tool: {e}")
            self.initialized = False

    def query(self, query_str: str) -> Optional[RagResponse]:
        if not self.initialized:
            logger.error("RAG Agent Tool is not initialized.")
            return None
            
        try:
            nodes = self.retriever.retrieve(query_str)
            context_str = "\n\n".join([n.get_content() for n in nodes])
            source_filenames = list(set([n.metadata.get("filename", "unknown") for n in nodes]))
            
            prompt_template_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "answer the query: {query_str}\n\n"
                "IMPORTANT INSTRUCTIONS:\n"
                "1. Extract ALL actionable tasks mentioned in the context, no matter how many (could be 0, could be 20+)\n"
                "2. Generate 2-5 key insights that capture the most important findings\n"
                "3. Provide a comprehensive summary\n"
                "4. Do NOT limit yourself to a fixed number of tasks - extract every single task you find\n"
                "5. Provide all source URLs in a proper format (e.g., [Title](URL)) if available in the context.\n"
            )
            
            program = OpenAIPydanticProgram.from_defaults(
                output_cls=RagResponse,
                prompt_template_str=prompt_template_str,
                llm=Settings.llm
            )
            
            response_obj = program(context_str=context_str, query_str=query_str)
            response_obj.sources = source_filenames
            
            return response_obj
            
        except Exception as e:
            logger.error(f"Error querying RAG Agent: {e}")
            return None
