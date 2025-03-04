from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import uvicorn
import os
import logging
from contextlib import asynccontextmanager

from model import ModelManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Model initialization
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    try:
        logger.info("Loading Deepseek model...")
        model_manager.load_model()
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
    yield
    # Clean up resources on shutdown
    try:
        logger.info("Unloading model...")
        model_manager.unload_model()
    except Exception as e:
        logging.error(f"Failed to unload model: {str(e)}")

app = FastAPI(lifespan=lifespan, title="Deepseek Model API", 
              description="API for generating text using the Deepseek LLM")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False

class GenerationResponse(BaseModel):
    text: str
    usage: Dict[str, int]

@app.get("/")
async def root():
    return {"message": "DeepSeek Model API", "status": "running", "model_loaded": model_manager.is_loaded()}

@app.get("/health")
async def health_check():
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logger.info(f"Received generation request with prompt: {request.prompt[:20]}...")
    logger.info(f"Generation parameters: max_tokens={request.max_tokens}, temp={request.temperature}, top_p={request.top_p}, top_k={request.top_k}")
    
    try:
        generated_text, usage = model_manager.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=request.stop_sequences,
        )
        
        return GenerationResponse(
            text=generated_text,
            usage=usage
        )
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Stream endpoint would be implemented here, but requires more complex SSE handling

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Significantly increase timeouts to prevent 502 errors
    uvicorn.run("main:app", host="0.0.0.0", port=port, timeout_keep_alive=300, 
                timeout_graceful_shutdown=300, timeout_notify=300, log_level="info") 