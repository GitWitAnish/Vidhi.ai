"""
FAISS vector store for efficient similarity search.
Handles index creation, storage, and retrieval operations.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import faiss
from threading import Lock

from app.core.config import settings

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    Manages FAISS index for vector similarity search.
    Thread-safe with persistent storage support.
    """
    
    def __init__(self, index_path: Optional[Path] = None):
        """
        Initialize the FAISS store.
        
        Args:
            index_path: Path to store/load the FAISS index
        """
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.index_file = self.index_path / "legal_faiss.index"
        self.metadata_file = self.index_path / "legal_faiss_meta.json"
        
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = settings.EMBEDDING_DIMENSION
        
        # Thread lock for concurrent access
        self._lock = Lock()
        
        # Try to load existing index
        self._load_index()
    
    def _load_index(self) -> bool:
        """
        Load existing FAISS index and metadata from disk.
        
        Returns:
            True if index was loaded successfully, False otherwise
        """
        try:
            if self.index_file.exists() and self.metadata_file.exists():
                logger.info(f"Loading FAISS index from {self.index_file}")
                
                self.index = faiss.read_index(str(self.index_file))
                
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                
                logger.info(
                    f"Loaded FAISS index with {self.index.ntotal} vectors "
                    f"and {len(self.metadata)} metadata entries"
                )
                return True
            else:
                logger.info("No existing FAISS index found, creating new one")
                self._create_new_index()
                return False
                
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self._create_new_index()
            return False
    
    def _create_new_index(self):
        """Create a new empty FAISS index."""
        # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
        # For better performance with large datasets, consider IndexIVFFlat
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def save_index(self) -> bool:
        """
        Save the current index and metadata to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with self._lock:
                # Ensure directory exists
                self.index_path.mkdir(parents=True, exist_ok=True)
                
                # Save FAISS index
                faiss.write_index(self.index, str(self.index_file))
                
                # Save metadata
                with open(self.metadata_file, "w", encoding="utf-8") as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
                
                logger.info(
                    f"Saved FAISS index with {self.index.ntotal} vectors "
                    f"to {self.index_file}"
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            return False
    
    def add_embeddings(
        self, 
        embeddings: np.ndarray, 
        metadata_list: List[Dict[str, Any]],
        already_normalized: bool = False
    ) -> int:
        """
        Add embeddings and their metadata to the index.
        Matches scripts/vector.py behavior with conditional normalization.
        
        Args:
            embeddings: numpy array of shape (n, dimension)
            metadata_list: List of metadata dicts for each embedding
            already_normalized: If True, assumes embeddings are already normalized
                               If False, normalizes them (matches scripts logic)
            
        Returns:
            Number of embeddings added
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) doesn't match "
                f"metadata count ({len(metadata_list)})"
            )
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) doesn't match "
                f"index dimension ({self.dimension})"
            )
        
        with self._lock:
            # Conditional normalization matching scripts/vector.py behavior
            # If embeddings are NOT already normalized, normalize them
            if not already_normalized:
                faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            # Add metadata
            self.metadata.extend(metadata_list)
            
            logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
            
            return len(embeddings)
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the index.
        Matches scripts/vector.py search behavior with score filtering.
        
        Args:
            query_embedding: Query vector of shape (1, dimension) or (dimension,)
            top_k: Number of results to return
            min_score: Minimum similarity score threshold (default 0.3 like scripts)
            
        Returns:
            List of results with metadata and similarity scores
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty, no results to return")
            return []
        
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Fetch 2x top_k to account for filtering by score (matches scripts/vector.py)
        fetch_k = min(top_k * 2, self.index.ntotal)
        
        with self._lock:
            # Search
            distances, indices = self.index.search(query_embedding, fetch_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            # Filter by min_score (matches scripts/vector.py behavior)
            if float(dist) < min_score:
                continue
                
            result = {
                "rank": len(results) + 1,
                "similarity_score": float(dist),
                "index": int(idx),
                **self.metadata[idx]
            }
            results.append(result)
            
            # Stop once we have enough good results
            if len(results) >= top_k:
                break
        
        logger.debug(f"Search returned {len(results)} results (min_score threshold: {min_score})")
        return results
    
    def delete_document(self, document_name: str) -> int:
        """
        Delete all chunks belonging to a specific document.
        Note: FAISS doesn't support direct deletion, so we rebuild the index.
        
        Args:
            document_name: Name of the document to delete
            
        Returns:
            Number of chunks deleted
        """
        with self._lock:
            # Find indices to keep
            keep_indices = [
                i for i, meta in enumerate(self.metadata)
                if meta.get("document_name") != document_name
            ]
            
            deleted_count = len(self.metadata) - len(keep_indices)
            
            if deleted_count == 0:
                logger.info(f"No chunks found for document: {document_name}")
                return 0
            
            # Reconstruct vectors for remaining indices
            if keep_indices:
                remaining_vectors = np.vstack([
                    self.index.reconstruct(i) for i in keep_indices
                ])
                remaining_metadata = [self.metadata[i] for i in keep_indices]
                
                # Create new index
                self._create_new_index()
                
                # Add remaining vectors back
                self.index.add(remaining_vectors)
                self.metadata = remaining_metadata
            else:
                # All vectors deleted, create empty index
                self._create_new_index()
            
            logger.info(f"Deleted {deleted_count} chunks for document: {document_name}")
            return deleted_count
    
    def get_document_count(self) -> int:
        """Get the number of unique documents in the index."""
        documents = set(meta.get("document_name") for meta in self.metadata)
        return len(documents)
    
    def get_total_chunks(self) -> int:
        """Get the total number of chunks in the index."""
        return self.index.ntotal if self.index else 0
    
    def get_all_documents(self) -> List[str]:
        """Get list of all indexed document names."""
        return list(set(meta.get("document_name") for meta in self.metadata))
    
    def clear(self):
        """Clear the entire index."""
        with self._lock:
            self._create_new_index()
            logger.info("FAISS index cleared")
