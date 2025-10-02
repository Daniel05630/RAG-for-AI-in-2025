"""
FastAPI entrypoint for RAG application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes_chat import router as chat_router
from api.routes_files import router as files_router
from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="RAG Application",
    description="A Retrieval-Augmented Generation application with LangGraph agent",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(files_router, prefix="/api/v1", tags=["files"])

@app.get("/")
async def root():
    """Serve the HTML UI"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )