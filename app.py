import logging
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import CustomDocChatbot

# Configure console-only JSON logging with emojis
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Custom JSON log formatter for structured output
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_entry)

# Apply JSON formatter to console handler
for handler in logging.getLogger().handlers:
    handler.setFormatter(JSONFormatter())

# Initialize FastAPI app with descriptive title
app = FastAPI(title="Bright Solution's RAG Bot")

# Enable CORS for React frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://brightssolution.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot instance
try:
    chatbot = CustomDocChatbot()
    logger.info({"message": "🤖 Chatbot initialized successfully"})
except Exception as e:
    logger.critical({"message": f"❌ Failed to initialize chatbot: {str(e)}"})
    raise

# Define request model for /chat endpoint
class QueryRequest(BaseModel):
    """Pydantic model for validating chat query requests."""
    query: str

@app.get("/")
async def root():
    """Root endpoint returning a welcome message."""
    return {"message": "Hello, I am Bright Solution's AI Bot! 🤖"}

@app.post("/chat")
async def chat(request: QueryRequest):
    """
    Handle chat queries with rate limiting and caching.
    
    Args:
        request (QueryRequest): JSON payload with the user's query.
    
    Returns:
        dict: Response containing the chatbot's reply.
    
    Raises:
        HTTPException: If the query is invalid or processing fails.
    """
    try:
        response = await chatbot.query(request.query)
        logger.info({"message": f"💬 Query processed: {request.query} | Response: {response}"})
        return {"reply": response}
    except Exception as e:
        logger.error({"message": f"❌ API error: {str(e)}"})
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify LLM and vector store status.
    
    Returns:
        dict: Status indicating if the chatbot is operational.
    
    Raises:
        HTTPException: If critical components are not initialized.
    """
    try:
        if chatbot.qa_chain and chatbot.vector_db:
            logger.info({"message": "✅ Health check passed"})
            return {"status": "healthy", "details": "LLM and vector store operational"}
        raise Exception("Chatbot components not initialized")
    except Exception as e:
        logger.error({"message": f"❌ Health check failed: {str(e)}"})
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    await chatbot.shutdown()
    logger.info({"message": "🛑 Application shutdown gracefully"})

if __name__ == "__main__":
    import uvicorn
    logger.info({"message": "🚀 Starting FastAPI server on port 8000"})
    uvicorn.run(app, host="0.0.0.0", port=8000)