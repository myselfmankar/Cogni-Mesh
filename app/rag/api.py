import os
import chromadb
import logging
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
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
    key_insights: List[str] = Field(..., description="List of key insights derived")
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
        # 1. Setup Embedding (OpenAI - reliable!)
        logging.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        embed_model = OpenAIEmbedding(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
        Settings.embed_model = embed_model
        
        # 2. Setup LLM (OpenAI)
        from llama_index.llms.openai import OpenAI
        logging.info("Initializing OpenAI LLM...")
        llm = OpenAI(model="gpt-4o-mini", api_key=config.OPENAI_API_KEY, temperature=0.1)
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
                logging.info("✅ BM25 index loaded")
            else:
                logging.warning("⚠️  BM25 index not found. Run ingestion first.")
        
        # 6. Initialize Graph Service
        if config.USE_NEO4J:
            graph_service = GraphService()
        
        # 7. Setup Reranker
        reranker = None
        if config.USE_RERANKING and config.COHERE_API_KEY:
            try:
                reranker = CohereRerank(api_key=config.COHERE_API_KEY, top_n=5)
                logging.info("✅ Cohere reranker initialized")
            except Exception as e:
                logging.warning(f"⚠️  Reranker initialization failed: {e}")
        
        # 8. Create Hybrid Query Engine
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
                
                # Convert to NodeWithScore format
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
            
            # Step 5: Generate structured response
            json_prompt = (
                f"Answer the following query based on the context: {query_str}\n"
                f"{prerequisite_context}\n"
                "\nContext:\n" + context_str + "\n\n"
                "Provide the response in valid JSON format with the following keys:\n"
                "- summary: Detailed, comprehensive summary (at least 3-4 paragraphs of depth)\n"
                "- key_insights: List of 5-7 actionable insights\n"
                "- tasks: List of objects with keys 'title', 'description', 'status'. (infer status as 'Pending', 'In Progress', or 'Completed')\n"
                "- critique: Critical evaluation of the answer's quality and completeness based ONLY on the context provided.\n"
                "- faithfulness_score: float between 0.0 and 1.0 (1.0 = fully supported by context, < 0.5 = major hallucinations)\n" 
                "Ensure strict JSON compliance. Do NOT wrap the output in markdown."
            )
            
            # Use OpenAI's native JSON mode for reliability
            response = llm.complete(
                json_prompt,
                formatted=True, 
                # LlamaIndex OpenAI integration supports output parsing but here we force text
                # Ideally we'd use Pydantic program, but let's fix the string parsing first
            )
            
            # Parse response
            import json
            import re
            
            text = str(response)
            # Remove markdown code blocks if present
            clean_text = re.sub(r'```json\s*|\s*```', '', text).strip()
            
            try:
                # Try parsing cleaned text directly first
                data = json.loads(clean_text)
            except json.JSONDecodeError:
                # Fallback to regex extraction if direct parse fails
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                    except:
                        data = {}
                else:
                    data = {}
            
            if data:
                tasks_data = data.get("tasks", [])
                formatted_tasks = []
                for t in tasks_data:
                    # Handle flexible schema from LLM
                    title = t.get("title") or t.get("task") or "Untitled Task"
                    description = t.get("description") or t.get("summary") or title
                    status = t.get("status") or "Pending"
                    formatted_tasks.append(Task(title=title, description=description, status=status))

                return RagResponse(
                    summary=data.get("summary", ""),
                    key_insights=data.get("key_insights", []),
                    tasks=formatted_tasks,
                    faithfulness_score=data.get("faithfulness_score", 0.0),
                    critique=data.get("critique", "Self-correction failed."),
                    sources=source_filenames
                )
            
            # Fallback
            return RagResponse(
                summary=text[:500] + "...",
                key_insights=["Could not parse structured insights."],
                tasks=[],
                faithfulness_score=0.1,
                critique="Failed to parse structured response. Low confidence.",
                sources=source_filenames
            )

        app.state.custom_query_fn = custom_query
        logging.info("🚀 Advanced RAG System Ready (Hybrid + Rerank + Graph)")
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
