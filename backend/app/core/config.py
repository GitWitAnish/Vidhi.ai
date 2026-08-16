
import os
from pydantic_settings import BaseSettings
from pathlib import Path
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App settings
    APP_NAME: str = "Vidhi.ai RAG API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parents[3]  # Project root
    DATA_DIR: Path = BASE_DIR / "backend" / "database"
    FAISS_INDEX_PATH: Path = DATA_DIR
    DOCUMENTS_PATH: Path = BASE_DIR / "dataset" / "processed"
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # LLM settings
    LLM_MODEL: str = "meta-llama/Llama-2-7b-chat-hf"
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    
    # RAG settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    
    # HuggingFace API settings
    HUGGINGFACE_API_KEY: str = ""
    HF_TOKEN: str = ""
    
    # CORS settings
    FRONTEND_URL: str = ""  # Set in Render env vars (e.g. https://vidhi-ai-frontend.onrender.com)
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173", "https://team-pdf.vercel.app"]
    
    class Config:
        env_file = str(Path(__file__).resolve().parents[3] / "backend" / ".env")
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Using lru_cache ensures settings are only loaded once.
    """
    return Settings()


# Global settings instance
settings = get_settings()

# Export HuggingFace API key to environment so os.getenv() picks it up
if settings.HUGGINGFACE_API_KEY:
    os.environ.setdefault("HUGGINGFACE_API_KEY", settings.HUGGINGFACE_API_KEY)
if settings.HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", settings.HF_TOKEN)

# Create necessary directories on startup
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
settings.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)

# Add deployed frontend URL to CORS origins
if settings.FRONTEND_URL and settings.FRONTEND_URL not in settings.CORS_ORIGINS:
    settings.CORS_ORIGINS.append(settings.FRONTEND_URL)
