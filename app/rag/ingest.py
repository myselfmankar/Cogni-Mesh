import os
import shutil
import logging
import time
import pickle
import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from rank_bm25 import BM25Okapi
from . import config
from .extractor import ContentExtractor
from .graph_service import GraphService

def ingest_documents(reset_db: bool = False) -> dict:
    stats = {"new_documents": 0, "status": "success", "error": None}

    embed_model = GeminiEmbedding(
        model_name=config.EMBEDDING_MODEL,
        api_key=config.GOOGLE_API_KEY,
        embed_batch_size=100  # CRITICAL FIX for Free Tier RPM limits
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
        return {"new_documents": 0, "status": "no_docs_dir"}

    logging.info(f"Scanning directory: {config.DOCS_DIR}")
    files = []
    for root, dirs, filenames in os.walk(config.DOCS_DIR):
        # Skip hidden subdirectories (like .cache) so we don't double-ingest cached markdown
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for filename in filenames:
            if not filename.startswith('.'):
                files.append(os.path.join(root, filename))

    for file_path in files:
        filename = os.path.basename(file_path)
        if filename in existing_files:
            logging.info(f"Skipping (already indexed): {filename}")
            continue

        logging.info(f"Processing: {file_path}")
        data = extractor.extract(file_path)
        if data and data.get("text", "").strip():
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
        return {"new_documents": 0, "status": "no_new_docs"}
    
    # 3.1 Build BM25 Index (for hybrid search)
    if config.USE_HYBRID_SEARCH:
        logging.info("Building BM25 index for hybrid search...")
        try:
            tokenized_corpus = [doc.text.lower().split() for doc in documents]
            # Ensure at least one token exists across all docs to avoid BM25 ZeroDivisionError
            if any(len(tokens) > 0 for tokens in tokenized_corpus):
                bm25_index = BM25Okapi(tokenized_corpus)
                
                # Save BM25 index
                bm25_path = os.path.join(config.BASE_DIR, "bm25_index.pkl")
                with open(bm25_path, "wb") as f:
                    pickle.dump({"bm25": bm25_index, "documents": documents}, f)
                logging.info(f"BM25 index saved to {bm25_path}")
            else:
                logging.warning("No valid text tokens found; skipping BM25 index generation.")
        except Exception as e:
            logging.error(f"Error building BM25 index: {e}")
            stats["error"] = str(e)
    
    # 3.2 Extract topics for Neo4j Knowledge Graph
    if graph_service:
        try:
            logging.info("Extracting topics and relationships for Knowledge Graph...")
            for doc in documents[:5]:  # Limit for speed
                filename = doc.metadata.get("filename", "unknown")
                graph_service.extract_topics_and_relations(doc.text, filename)
            logging.info("Knowledge graph populated")
        except Exception as e:
            logging.error(f"Graph population failed: {e}")
            stats["error"] = str(e)

    logging.info(f"Ingesting {len(documents)} new documents into ChromaDB...")
    logging.info(f"Processing in batches of {config.EMBEDDING_BATCH_SIZE} to avoid rate limits...")
    
    # 4. Create Index with Batch Processing and Retry Logic
    # Process documents in batches to avoid rate limits
    failed_docs = 0
    batch_size = max(1, config.EMBEDDING_BATCH_SIZE)
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(documents))
        batch_docs = documents[start_idx:end_idx]
        
        retry_count = 0
        delay = config.RETRY_INITIAL_DELAY
        quota_exceeded = False
        
        while retry_count < config.RETRY_MAX_ATTEMPTS:
            try:
                logging.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_docs)} documents)...")
                
                # Create/update index with current batch
                if batch_num == 0 and len(existing_files) == 0:
                    index = VectorStoreIndex.from_documents(
                        batch_docs, storage_context=storage_context
                    )
                else:
                    for doc in batch_docs:
                        index.insert(doc)
                
                logging.info(f"Batch {batch_num + 1}/{total_batches} completed")
                break   
                
            except Exception as e:
                if "429" in str(e) or "Resource has been exhausted" in str(e) or "quota" in str(e).lower():
                    logging.error("QUOTA EXCEEDED: Detected 429 error. Breaking instantly.")
                    quota_exceeded = True
                    failed_docs += len(batch_docs)
                    stats["error"] = str(e)
                    break # Break retry loop immediately without sleeping
                    
                retry_count += 1
                logging.error(f"Error processing batch {batch_num + 1}: {e}")
                
                if retry_count >= config.RETRY_MAX_ATTEMPTS:
                    logging.error(f"Max retries reached for batch {batch_num + 1}.")
                    failed_docs += len(batch_docs)
                    stats["error"] = str(e)
                    break
                
                logging.warning(f"Retrying after {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, config.RETRY_MAX_DELAY)  # Exponential backoff
        
        if quota_exceeded:
            logging.error("Stopping ingestion due to persistent Quota Exceeded error.")
            # Calculate remaining un-attempted docs
            remaining_docs = len(documents) - end_idx
            failed_docs += remaining_docs
            break
            
        # Mandatory delay between batches to stay within free-tier RPM
        if batch_num < total_batches - 1:
            logging.info(f"Cooling down for {config.BATCH_DELAY}s...")
            time.sleep(config.BATCH_DELAY)
    
    logging.info("Ingestion complete!")
    
    # Cleanup
    if graph_service:
        graph_service.close()
    
    final_status = "success" if failed_docs == 0 else "partial_failure"
    if failed_docs == len(documents):
        final_status = "failed"
        
    return {
        "new_documents": len(documents) - failed_docs, 
        "failed_documents": failed_docs,
        "status": final_status,
        "error": stats["error"]
    }

if __name__ == "__main__":
    ingest_documents()
