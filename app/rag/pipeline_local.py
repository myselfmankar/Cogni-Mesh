import asyncio
import logging
import sys
import os
import shutil
from src.rag.tool import RagAgentTool
from src.video.video_gen import VideoGenerator
from src.state import VideoGenerationState

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global configuration for local copy
COPY_LOCAL = False  # Set to True to copy final video to output_videos folder

def copy_to_local(video_path: str, output_folder: str = "output_videos") -> str:
    """
    Copy video from temp location to local output folder.
    
    Args:
        video_path: Path to the generated video
        output_folder: Destination folder name (default: "output_videos")
    
    Returns:
        Path to the copied video file
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return video_path
    
    # Create output directory if it doesn't exist
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(current_dir, output_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the video
    filename = os.path.basename(video_path)
    dest_path = os.path.join(output_dir, filename)
    
    try:
        shutil.copy2(video_path, dest_path)
        logger.info(f"Copied video to local folder: {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to copy video to local folder: {e}")
        return video_path

async def run_pipeline(query_text: str = None, state: VideoGenerationState = None):
    """
    Run the RAG and video generation pipeline.
    
    Args:
        query_text: Query string (used if state is not provided)
        state: VideoGenerationState object (preferred method)
    
    Returns:
        Updated VideoGenerationState if state was provided, otherwise None
    """
    # Determine query source
    if state:
        query = state.topic
        logger.info(f"STARTING PIPELINE WITH STATE | Query: {query}")
        
    elif query_text:
        query = query_text
        
        logger.info(f"STARTING PIPELINE | Query: {query}")
        
    else:
        logger.error("No query provided. Either query_text or state must be specified.")
        return None

    # Use VideoGenerator's main method if state is provided
    if state:
        generator = VideoGenerator()
        try:
            updated_state = await generator.main(state)
            
            # Copy to local if COPY_LOCAL is enabled
            if COPY_LOCAL and updated_state.slide_video_path:
                # Extract local path from output_videos
                video_filename = "slide_video.mp4"
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                local_video_path = os.path.join(current_dir, "output_videos", video_filename)
                
                if os.path.exists(local_video_path):
                    copy_to_local(local_video_path)
            
            
            logger.info(f"SUCCESS! Pipeline completed")
            logger.info(f"Session ID: {updated_state.session_id}")
            logger.info(f"Status: {updated_state.status}")
            
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if state:
                state.set_error(f"Pipeline error: {str(e)}")
                return state
            return None
    
    # Legacy path: Direct RAG + Video generation (no state)
    else:
        # 1. Initialize & Query RAG Tool
        rag_tool = RagAgentTool()
        
        logger.info("Querying Knowledge Base...")
        rag_data = rag_tool.query(query)
        
        if not rag_data:
            logger.error("Failed to get data from RAG Tool. Aborting.")
            return

        logger.info("RAG Data Retrieved:")
        logger.info(f"- Summary: {rag_data.summary[:50]}...")
        logger.info(f"- Insights: {len(rag_data.key_insights)}")
        logger.info(f"- Tasks: {len(rag_data.tasks)}")

        # 2. Generate Video
        logger.info("Starting Video Generation Node...")
        generator = VideoGenerator()
        try:
            output_path = await generator.generate_video_from_rag(rag_data, query, filename="rag_report_tool.mp4")
            
            # Copy to local if enabled
            if COPY_LOCAL and output_path:
                copy_to_local(output_path)
            
            
            logger.info(f"SUCCESS! Video saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Video Generation failed: {e}")

if __name__ == "__main__":
    query = "Summarize the key tasks and insights."
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    asyncio.run(run_pipeline(query_text=query))

