from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from immigration_agents_team import ImmigrationGPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the ImmigrationGPT instance
immigration_gpt_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application lifespan."""
    global immigration_gpt_instance
    try:
        logger.info("Starting ImmigrationGPT backend...")
        # Change to parent directory to find data folder
        original_dir = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(parent_dir)
        
        immigration_gpt_instance = ImmigrationGPT()
        logger.info("ImmigrationGPT backend started successfully!")
        yield
    except Exception as e:
        logger.error(f"Failed to start ImmigrationGPT backend: {e}")
        raise
    finally:
        logger.info("Shutting down ImmigrationGPT backend...")
        if 'original_dir' in locals():
            os.chdir(original_dir)

# Create FastAPI app
app = FastAPI(
    title="ImmigrationGPT API",
    description="AI-powered immigration policy assistant with RAG and web search capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str
    sources: Optional[List[str]] = None
    tool_used: Optional[str] = None
    tool_description: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

class SettingsRequest(BaseModel):
    openai_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    langchain_api_key: Optional[str] = None

class SettingsResponse(BaseModel):
    status: str
    message: str

# Dependency to get the ImmigrationGPT instance
def get_immigration_gpt() -> ImmigrationGPT:
    if immigration_gpt_instance is None:
        raise HTTPException(status_code=503, detail="ImmigrationGPT service not available")
    return immigration_gpt_instance

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    return HealthResponse(
        status="healthy",
        message="ImmigrationGPT API is running",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        if immigration_gpt_instance is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        return HealthResponse(
            status="healthy",
            message="ImmigrationGPT API is running",
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    immigration_gpt: ImmigrationGPT = Depends(get_immigration_gpt)
):
    """Process a chat message and return a response."""
    try:
        logger.info(f"Processing chat request: {request.message[:100]}...")
        
        # Check if the system is ready
        if not immigration_gpt.is_ready():
            return ChatResponse(
                response="The system is not ready yet. Please configure your API keys in the settings to start using ImmigrationGPT.",
                conversation_id=request.conversation_id or f"conv_{hash(request.message) % 1000000}",
                timestamp=datetime.now().isoformat(),
                sources=None,
                tool_used="None",
                tool_description="System not ready - API keys required"
            )
        
        # Process the message through ImmigrationGPT
        result = immigration_gpt.ask(request.message)
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or f"conv_{hash(request.message) % 1000000}"
        
        return ChatResponse(
            response=result["response"],
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            sources=None,  # Could be enhanced to return source documents
            tool_used=result.get("tool_used"),
            tool_description=result.get("tool_description")
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/settings", response_model=SettingsResponse)
async def update_settings(request: SettingsRequest):
    """Update API key settings."""
    try:
        updated_keys = []
        
        if request.openai_api_key:
            os.environ["OPENAI_API_KEY"] = request.openai_api_key
            updated_keys.append("OpenAI")
        
        if request.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = request.tavily_api_key
            updated_keys.append("Tavily")
        
        if request.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = request.langchain_api_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            updated_keys.append("LangChain")
        
        if updated_keys:
            # Reinitialize the ImmigrationGPT instance with new keys
            global immigration_gpt_instance
            try:
                immigration_gpt_instance = ImmigrationGPT()
                if immigration_gpt_instance.is_ready():
                    logger.info(f"✅ Reinitialized ImmigrationGPT with updated keys: {', '.join(updated_keys)}")
                else:
                    logger.warning(f"⚠️ ImmigrationGPT initialized but not ready. Missing required API keys.")
            except Exception as e:
                logger.error(f"❌ Failed to reinitialize with new keys: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid API keys: {str(e)}")
        
        return SettingsResponse(
            status="success",
            message=f"Updated settings for: {', '.join(updated_keys) if updated_keys else 'No changes'}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating settings: {str(e)}")

@app.get("/settings/status")
async def get_settings_status():
    """Get the current status of API key settings."""
    try:
        status = {
            "openai_api_key": "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not set",
            "tavily_api_key": "✅ Set" if os.getenv("TAVILY_API_KEY") else "❌ Not set",
            "langchain_api_key": "✅ Set" if os.getenv("LANGCHAIN_API_KEY") else "❌ Not set (optional)",
            "service_status": "✅ Ready" if immigration_gpt_instance and immigration_gpt_instance.is_ready() else "❌ Not ready - API keys required"
        }
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting settings status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting settings status: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
