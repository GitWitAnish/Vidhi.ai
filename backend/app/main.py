"""
FastAPI application entry point.
Configures the app, middleware, and routes.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routes import upload_router, ask_router, health_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Data directory: {settings.DATA_DIR}")
    
    # Pre-load services (optional - comment out for faster startup)
    # This ensures models are loaded before first request
    try:
        from app.services.rag_service import RAGService
        rag_service = RAGService()
        stats = rag_service.get_index_stats()
        logger.info(f"FAISS index loaded: {stats['total_chunks']} chunks from {stats['total_documents']} documents")
    except Exception as e:
        logger.warning(f"Failed to pre-load services: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Vidhi.ai RAG API
    
    A RAG (Retrieval-Augmented Generation) API for querying Nepali legal documents.
    
    ### Features:
    - **Document Upload**: Upload PDF documents to be indexed
    - **Question Answering**: Ask questions and get AI-generated answers
    - **Source Citations**: See which documents were used for the answer
    
    ### How it works:
    1. Upload PDF documents using `/upload`
    2. Documents are processed, chunked, and embedded
    3. Ask questions using `/ask`
    4. The system retrieves relevant chunks and generates answers using LLaMA
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)  # Includes /, /health, /stats
app.include_router(upload_router)  # /upload
app.include_router(ask_router)     # /ask


# Run with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
