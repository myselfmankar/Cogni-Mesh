import os
import shutil
import logging
import time
import pickle
import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from rank_bm25 import BM25Okapi
from . import config
from .extractor import ContentExtractor
from .graph_service import GraphService

def ingest_documents(reset_db: bool = False):
    # 1. Setup Embedding Model (OpenAI - reliable!)
    logging.info(f"Initializing embedding model: {config.EMBEDDING_MODEL}")
    embed_model = OpenAIEmbedding(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY
    )
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    # 2. Setup ChromaDB
    if not os.path.exists(config.CHROMA_DB_DIR):
        os.makedirs(config.CHROMA_DB_DIR)

    if reset_db:
        if os.path.exists(config.CHROMA_DB_DIR):
            logging.info(f"Resetting database at {config.CHROMA_DB_DIR}...")
            shutil.rmtree(config.CHROMA_DB_DIR)
            os.makedirs(config.CHROMA_DB_DIR)
    
    # Re-initialize client after deletion or if persisting
    db = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)

    chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2.1 Check existing files for incremental update
    existing_files = set()
    if not reset_db:
        # Get all metadata to check what's already indexed
        # Check if collection is not empty
        if chroma_collection.count() > 0:
            result = chroma_collection.get(include=["metadatas"])
            if result and result.get("metadatas"):
                for meta in result["metadatas"]:
                    if meta and "filename" in meta:
                        existing_files.add(meta["filename"])
            logging.info(f"Found {len(existing_files)} files already in index.")

    # 3. Initialize Graph Service (Neo4j)
    graph_service = GraphService() if config.USE_NEO4J else None
    
    # 4. Extract Documents
    extractor = ContentExtractor(whisper_model_size=config.WHISPER_MODEL_SIZE)
    documents = []
    
    if not os.path.exists(config.DOCS_DIR):
        logging.warning(f"Documents directory not found: {config.DOCS_DIR}")
        logging.info("Please create it and add files.")
        return

    logging.info(f"Scanning directory: {config.DOCS_DIR}")
    files = []
    for root, _, filenames in os.walk(config.DOCS_DIR):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    for file_path in files:
        filename = os.path.basename(file_path)
        if filename in existing_files:
            logging.info(f"Skipping (already indexed): {filename}")
            continue

        logging.info(f"Processing: {file_path}")
        data = extractor.extract(file_path)
        if data and data.get("text"):
            # Create LlamaIndex Document
            # Use filename as ID or letting LlamaIndex generate one
            # Storing filename in metadata is crucial for citations
            metadata = data.get("metadata", {})
            metadata["filename"] = filename
            metadata["path"] = file_path # Absolute path
            
            doc = Document(
                text=data["text"],
                metadata=metadata,
                excluded_llm_metadata_keys=["path"], # Don't send path to LLM if context is tight
                excluded_embed_metadata_keys=["path"]
            )
            documents.append(doc)
    
    if not documents:
        logging.warning("No new documents to ingest.")
        if graph_service:
            graph_service.close()
        return
    
    # 3.1 Build BM25 Index (for hybrid search)
    if config.USE_HYBRID_SEARCH:
        logging.info("Building BM25 index for hybrid search...")
        tokenized_corpus = [doc.text.lower().split() for doc in documents]
        bm25_index = BM25Okapi(tokenized_corpus)
        
        # Save BM25 index
        bm25_path = os.path.join(config.BASE_DIR, "bm25_index.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25_index, "documents": documents}, f)
        logging.info(f"✅ BM25 index saved to {bm25_path}")
    
    # 3.2 Extract topics for Neo4j Knowledge Graph
    if graph_service:
        logging.info("Extracting topics and relationships for Knowledge Graph...")
        for doc in documents[:5]:  # Limit to first 5 docs for hackathon speed
            filename = doc.metadata.get("filename", "unknown")
            graph_service.extract_topics_and_relations(doc.text, filename)
        logging.info("✅ Knowledge graph populated")

    logging.info(f"Ingesting {len(documents)} new documents into ChromaDB...")
    logging.info(f"Processing in batches of {config.EMBEDDING_BATCH_SIZE} to avoid rate limits...")
    
    # 4. Create Index with Batch Processing and Retry Logic
    # Process documents in batches to avoid rate limits
    batch_size = config.EMBEDDING_BATCH_SIZE
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(documents))
        batch_docs = documents[start_idx:end_idx]
        
        retry_count = 0
        delay = config.RETRY_INITIAL_DELAY
        
        while retry_count < config.RETRY_MAX_ATTEMPTS:
            try:
                logging.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_docs)} documents)...")
                
                # Create/update index with current batch
                if batch_num == 0:
                    # First batch: create new index
                    index = VectorStoreIndex.from_documents(
                        batch_docs, text_splitters=None, storage_context=storage_context
                    )
                else:
                    # Subsequent batches: add to existing index
                    for doc in batch_docs:
                        index.insert(doc)
                
                logging.info(f"✅ Batch {batch_num + 1}/{total_batches} completed")
                break  # Success, move to next batch
                
            except Exception as e:
                retry_count += 1
                logging.error(f"❌ Error processing batch {batch_num + 1}: {e}")
                if retry_count >= config.RETRY_MAX_ATTEMPTS:
                    logging.error(f"❌ Max retries reached for batch {batch_num + 1}. Skipping...")
                    break
                
                logging.warning(f"⚠️  Retrying after {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, config.RETRY_MAX_DELAY)  # Exponential backoff
        
        # Small delay between batches to be respectful of rate limits
        if batch_num < total_batches - 1:
            time.sleep(0.5)
    
    logging.info("Ingestion complete!")
    
    # Cleanup
    if graph_service:
        graph_service.close()

if __name__ == "__main__":
    ingest_documents()
