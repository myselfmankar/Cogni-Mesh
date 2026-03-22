import logging
import chromadb
from typing import List, Optional
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.program import LLMTextCompletionProgram
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
    step_by_step_solution: Optional[str] = Field(None, description="Detailed pedagogical step-by-step solution for the query")
    key_insights: List[str] = Field(
        ..., 
        description="List of 2-5 key insights derived from the content. Extract the most important findings."
    )
    learning_objectives: List[str] = Field(default_factory=list, description="What the student should learn from this answer")
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
            
            embed_model = GeminiEmbedding(
                model_name=config.EMBEDDING_MODEL,
                api_key=config.GOOGLE_API_KEY
            )
            Settings.embed_model = embed_model
            
            Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=config.GOOGLE_API_KEY, temperature=0)

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
                "You are a Stellar Study Agent. Your goal is to help a student master their subjects and solve PYQs (Previous Year Questions) with precision.\n"
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "answer the query: {query_str}\n\n"
                "IMPORTANT INSTRUCTIONS:\n"
                "1. If the query is a question or problem (like a PYQ), provide a clear, pedagogical step-by-step solution in 'step_by_step_solution'.\n"
                "2. Generate 2-5 key insights for exam preparation.\n"
                "3. List 3-5 learning objectives that capture what the student will learn.\n"
                "4. Extract ALL actionable tasks (e.g., 'Solve 5 practice problems').\n"
                "5. Provide a comprehensive summary of core concepts.\n"
            )
            
            program = LLMTextCompletionProgram.from_defaults(
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
