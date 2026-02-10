
import uvicorn
import os
import shutil
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    print("\n" + "="*50)
    print("   Cogni-Mesh: Autonomous Intelligence Platform")
    print("   Powered by Gemini 1.5 & LlamaIndex")
    print("="*50 + "\n")

    # 1. Environment Check
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY is missing!")
        print("   -> Please create a .env file with GOOGLE_API_KEY=...")
        return

    # 2. Database Check
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    if not os.path.exists(db_path) or not os.listdir(db_path):
        print("⚠️  WARNING: Vector Database not found at ./chroma_db")
        print("   -> Ideally, run ingestion first: python -m app.rag.ingest")
        print("   -> The system will start, but queries might return empty results.")
    else:
        print("✅ Vector Database found.")

    # 3. Start Server
    print("\n🚀 Starting RAG API Server...")
    print("   -> Swagger Docs: http://127.0.0.1:8000/docs")
    print("   -> RAG Endpoint: POST http://127.0.0.1:8000/rag-query")
    print("\nPress Ctrl+C to stop.\n")
    
    # Run the "Better RAG System" (app.rag.api)
    uvicorn.run("app.rag.api:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
