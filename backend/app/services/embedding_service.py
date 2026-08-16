"""
Embedding service for converting text to vector representations.
Uses sentence-transformers for high-quality embeddings.
"""

import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Handles text-to-embedding conversion using sentence-transformers.
    Implements singleton pattern for efficient model loading.
    Matches scripts/embedding.py behavior with conditional normalization.
    """
    
    _instance = None
    _model = None
    _normalize = True  # Default normalization flag
    
    def __new__(cls):
        """Singleton pattern to ensure model is loaded only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, normalize: bool = True):
        """Initialize the embedding model.
        
        Args:
            normalize: Whether to normalize embeddings (default True)
        """
        if EmbeddingService._model is None:
            EmbeddingService._normalize = normalize
            self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformer model."""
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            EmbeddingService._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(
                f"Embedding model loaded successfully. "
                f"Dimension: {self.get_dimension()}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the loaded model."""
        if EmbeddingService._model is None:
            self._load_model()
        return EmbeddingService._model
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert a single text to embedding vector.
        Uses conditional normalization matching scripts/embedding.py.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=EmbeddingService._normalize,  # Conditional
            show_progress_bar=False
        )
        
        return embedding
    
    def embed_texts(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Convert multiple texts to embedding vectors.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings with shape (n_texts, dimension)
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if len(valid_texts) != len(texts):
            logger.warning(
                f"Filtered out {len(texts) - len(valid_texts)} empty texts"
            )
        
        if not valid_texts:
            raise ValueError("All texts are empty")
        
        logger.info(f"Embedding {len(valid_texts)} texts...")
        
        embeddings = self.model.encode(
            valid_texts,
            convert_to_numpy=True,
            normalize_embeddings=EmbeddingService._normalize,  # Conditional (matches scripts/embedding.py)
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query.
        Same as embed_text but semantically clearer for search operations.
        
        Args:
            query: Search query to embed
            
        Returns:
            Query embedding vector
        """
        return self.embed_text(query)
