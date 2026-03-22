import os
import chromadb
import logging
import pickle
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from rank_bm25 import BM25Okapi
from . import config
from .graph_service import GraphService

app = FastAPI(title="RAG Prototype API")

# Models for structured output
class Task(BaseModel):
    """
    Represents a single actionable task extracted from the context.
    """
    title: str = Field(..., description="Title of the task")
    description: str = Field(..., description="Detailed description")
    status: str = Field(..., description="Status (Pending, In Progress, Done)")

class RagResponse(BaseModel):
    """
    Structured response containing summary, insights, and actionable tasks derived from the knowledge base.
    """
    summary: str = Field(..., description="Comprehensive summary of the matched content")
    step_by_step_solution: Optional[str] = Field(None, description="Detailed pedagogical step-by-step solution for the query")
    key_insights: List[str] = Field(..., description="List of key insights derived")
    learning_objectives: List[str] = Field(default_factory=list, description="What the student should learn from this answer")
    tasks: List[Task] = Field(..., description="Actionable tasks identified")
    sources: List[str] = Field(..., description="List of source filenames used")
    faithfulness_score: float = Field(..., description="Self-evaluated score (0.0-1.0) of how grounded the answer is in the context")
    critique: str = Field(..., description="Internal Critic's evaluation of the answer")

class QueryRequest(BaseModel):
    query: str

from contextlib import asynccontextmanager

# Global state
query_engine = None
bm25_data = None
graph_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global query_engine, bm25_data, graph_service
    
    try:
        # 1. Setup Embedding (Gemini)
        embed_model = GeminiEmbedding(model_name=config.EMBEDDING_MODEL, api_key=config.GOOGLE_API_KEY)
        Settings.embed_model = embed_model
        
        # 2. Setup LLM (Gemini)
        llm = Gemini(model="models/gemini-2.5-flash", api_key=config.GOOGLE_API_KEY, temperature=0.1)
        Settings.llm = llm

        # 3. Connect to ChromaDB
        logging.info(f"Connecting to ChromaDB at {config.CHROMA_DB_DIR}")
        db = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # 4. Load Vector Index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        
        # 5. Load BM25 Index (for hybrid search)
        if config.USE_HYBRID_SEARCH:
            bm25_path = os.path.join(config.BASE_DIR, "bm25_index.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    bm25_data = pickle.load(f)
                logging.info("BM25 index loaded")
            else:
                logging.warning("BM25 index not found. Run ingestion first.")
        
        # 6. Initialize Graph Service
        if config.USE_NEO4J:
            graph_service = GraphService()
        
        # 7. Setup Reranker
        reranker = None
        if config.USE_RERANKING and config.COHERE_API_KEY:
            try:
                reranker = CohereRerank(api_key=config.COHERE_API_KEY, top_n=5)
                logging.info("Cohere reranker initialized")
            except Exception as e:
                logging.warning(f"Reranker initialization failed: {e}")        # 8. Create Hybrid Query Engine
        def custom_query(query_str: str) -> RagResponse:
            # Step 1: Check for knowledge gaps (Neo4j)
            prerequisite_context = ""
            if graph_service:
                gaps = graph_service.check_knowledge_gaps(query_str)
                if gaps["missing_prerequisites"]:
                    prerequisite_context = f"\n\nNote: This query involves {', '.join(gaps['mentioned_topics'])}. Prerequisites you should know: {', '.join(gaps['missing_prerequisites'])}."
            
            # Step 2: Hybrid Retrieval
            nodes = []
            
            # 2a. Vector retrieval
            vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
            vector_nodes = vector_retriever.retrieve(query_str)
            nodes.extend(vector_nodes)
            
            # 2b. BM25 retrieval (if enabled)
            if bm25_data:
                query_tokens = query_str.lower().split()
                bm25_scores = bm25_data["bm25"].get_scores(query_tokens)
                top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
                
                from llama_index.core.schema import NodeWithScore, TextNode
                for idx in top_bm25_indices:
                    doc = bm25_data["documents"][idx]
                    node = TextNode(text=doc.text, metadata=doc.metadata)
                    nodes.append(NodeWithScore(node=node, score=float(bm25_scores[idx])))
            
            # Step 3: Reranking
            if reranker:
                nodes = reranker.postprocess_nodes(nodes, query_str=query_str)
            
            # Step 4: Extract context and sources
            context_str = "\n\n".join([n.node.get_content() for n in nodes[:5]])
            source_filenames = list(set([n.node.metadata.get("filename", "unknown") for n in nodes[:5]]))
            
            # Step 5: Structured Generation using LlamaIndex Program
            prompt_template_str = (
                "You are a Stellar Study Agent. Your goal is to help a student master their subjects and solve PYQs (Previous Year Questions) with precision.\n"
                "Answer the following query based on the context: {query_str}\n"
                "{prerequisite_context}\n"
                "\nContext:\n" + "{context_str}" + "\n\n"
                "IMPORTANT:\n"
                "- summary: Detailed, comprehensive summary of the core concepts\n"
                "- step_by_step_solution: If the query is a question or problem, provide a clear, pedagogical step-by-step solution\n"
                "- key_insights: List of 5-7 actionable insights for exam preparation\n"
                "- learning_objectives: List what the student will learn from this explanation\n"
                "- tasks: List of tasks with 'title', 'description', 'status' (e.g., 'Solve 5 practice problems')\n"
                "- critique: Evaluation of the answer quality from a teacher's perspective\n"
                "- faithfulness_score: float (0.0-1.0) grounded strictly in context"
            )

            program = LLMTextCompletionProgram.from_defaults(
                output_cls=RagResponse,
                prompt_template_str=prompt_template_str,
                llm=Settings.llm
            )
            
            response_obj = program(
                context_str=context_str, 
                query_str=query_str,
                prerequisite_context=prerequisite_context
            )
            response_obj.sources = source_filenames
            
            return response_obj

        app.state.custom_query_fn = custom_query
        logging.info("Advanced RAG System Ready (Hybrid + Rerank + Graph)")
        
    except Exception as e:
        logging.error(f"Failed to initialize RAG system: {e}")
        
    yield
    # Cleanup code could go here

app = FastAPI(title="RAG Prototype API", lifespan=lifespan)

@app.get("/graph-data")
def get_graph_data(limit: int = 50):
    if not hasattr(app.state, 'custom_query_fn') or not graph_service:
         return {"nodes": [], "links": []}
    return graph_service.get_graph_data(limit)

@app.post("/rag-query", response_model=RagResponse)
def query_rag(request: QueryRequest):
    if not hasattr(app.state, 'custom_query_fn'):
         raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        response = app.state.custom_query_fn(request.query)
        return response
    except Exception as e:
        logging.error(f"Error processing query: {request.query} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload documents and trigger background ingestion"""
    from .ingest import ingest_documents
    
    if not os.path.exists(config.DOCS_DIR):
        os.makedirs(config.DOCS_DIR)
        
    uploaded_files = []
    for file in files:
        file_path = os.path.join(config.DOCS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file.filename)
    
    def run_ingestion():
        try:
            logging.info(f"Triggering auto-ingestion for uploaded files...")
            results = ingest_documents(reset_db=False)
            logging.info(f"Auto-ingestion completed: {results}")
        except Exception as e:
            logging.error(f"Auto-ingestion failed: {e}")

    background_tasks.add_task(run_ingestion)
    
    return {
        "message": f"Successfully uploaded {len(uploaded_files)} files. Ingestion started in background.", 
        "files": uploaded_files
    }

@app.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks, reset: bool = False):
    """Trigger the document ingestion process in the background"""
    from .ingest import ingest_documents
    
    def run_ingestion():
        try:
            logging.info("Starting background ingestion...")
            results = ingest_documents(reset_db=reset)
            logging.info(f"Background ingestion completed: {results}")
        except Exception as e:
            logging.error(f"Background ingestion failed: {e}")

    background_tasks.add_task(run_ingestion)
    return {"message": "Ingestion process started in the background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
