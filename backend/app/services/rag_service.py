"""
RAG (Retrieval-Augmented Generation) service.
Orchestrates the complete RAG pipeline: retrieval, prompt building, and generation.
"""

import logging
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path

from app.core.config import settings
from app.stores.faiss_store import FAISSStore
from app.utils.pdf_parser import PDFParser
from app.utils.text_chunker import TextChunker, TextChunk
from .embedding_service import EmbeddingService
from .llm_service import LLMService

logger = logging.getLogger(__name__)


class RAGService:
    """
    Main RAG service that coordinates document processing,
    1edding, retrieval, and answer generation.
    """
    
    def __init__(self):
        """Initialize all required services."""
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.faiss_store = FAISSStore()
        self.text_chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.pdf_parser = PDFParser()
    
    async def process_document(
        self, 
        file_content: bytes, 
        filename: str
    ) -> Dict[str, Any]:
        """
        Process an uploaded document: extract text, chunk, embed, and store.
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract text from PDF
            logger.info(f"Extracting text from: {filename}")
            text = self.pdf_parser.extract_text_from_bytes(file_content, filename)
            
            if not text or len(text.strip()) < 100:
                raise ValueError(
                    f"Insufficient text extracted from {filename}. "
                    f"The document may be scanned or image-based."
                )
            
            # Step 2: Chunk the text
            logger.info(f"Chunking text from: {filename}")
            chunks = self.text_chunker.chunk_text(text, filename)
            
            if not chunks:
                raise ValueError(f"No chunks created from {filename}")
            
            # Step 3: Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(chunk_texts)
            
            # Step 4: Prepare metadata
            metadata_list = [
                {
                    "text": chunk.text,
                    "document_name": filename,
                    "chunk_index": chunk.index,
                    "char_count": len(chunk.text)
                }
                for chunk in chunks
            ]
            
            # Step 5: Add to FAISS index
            logger.info(f"Adding {len(embeddings)} embeddings to FAISS")
            self.faiss_store.add_embeddings(embeddings, metadata_list)
            
            # Step 6: Save the index
            self.faiss_store.save_index()
            
            # Also save the original document
            doc_path = settings.DOCUMENTS_PATH / filename
            with open(doc_path, "wb") as f:
                f.write(file_content)
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Successfully processed {filename}: "
                f"{len(chunks)} chunks in {processing_time:.2f}s"
            )
            
            return {
                "success": True,
                "message": "Document processed and indexed successfully",
                "document_name": filename,
                "total_chunks": len(chunks),
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {filename}: {e}")
            raise
    
    async def search_and_answer(
        self, 
        question: str,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Search for relevant chunks and generate an answer.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and source chunks
        """
        start_time = time.time()
        top_k = top_k or settings.TOP_K_RESULTS
        
        try:
            # Step 1: Embed the question
            logger.info(f"Processing question: {question[:100]}...")
            query_embedding = self.embedding_service.embed_query(question)
            
            # Step 2: Search FAISS
            logger.info(f"Searching FAISS for top {top_k} results")
            search_results = self.faiss_store.search(query_embedding, top_k)
            
            if not search_results:
                return {
                    "answer": (
                        "I couldn't find any relevant information in the indexed documents. "
                        "Please make sure documents have been uploaded and try rephrasing your question."
                    ),
                    "sources": [],
                    "question": question,
                    "processing_time": round(time.time() - start_time, 2)
                }
            
            # Step 3: Extract context chunks
            context_chunks = [result["text"] for result in search_results]
            
            # Step 4: Generate answer (generate_answer builds the RAG prompt internally)
            logger.info("Generating answer with LLM")
            answer = self.llm_service.generate_answer(question, context_chunks)
            
            # Step 5: Format sources
            # Handle both metadata schemas:
            #   - Legacy (from scripts): {id, text, metadata: {filename, title, ...}}
            #   - New (from upload route): {text, document_name, chunk_index, ...}
            sources = []
            for result in search_results:
                meta = result.get("metadata", {})
                doc_name = (
                    result.get("document_name")
                    or meta.get("filename")
                    or meta.get("title")
                    or "Unknown"
                )
                chunk_idx = result.get("chunk_index", result.get("id", 0))
                text = result.get("text", "")
                sources.append({
                    "text": text[:500] + ("..." if len(text) > 500 else ""),
                    "document_name": doc_name,
                    "chunk_index": chunk_idx,
                    "similarity_score": round(result["similarity_score"], 4)
                })
            
            processing_time = time.time() - start_time
            
            logger.info(f"Generated answer in {processing_time:.2f}s")
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current FAISS index."""
        return {
            "total_documents": self.faiss_store.get_document_count(),
            "total_chunks": self.faiss_store.get_total_chunks(),
            "documents": self.faiss_store.get_all_documents(),
            "embedding_dimension": self.embedding_service.get_dimension(),
            "llm_loaded": self.llm_service.is_loaded()
        }
    
    def delete_document(self, document_name: str) -> int:
        """
        Delete a document from the index.
        
        Args:
            document_name: Name of the document to delete
            
        Returns:
            Number of chunks deleted
        """
        # Delete from FAISS
        deleted_count = self.faiss_store.delete_document(document_name)
        
        if deleted_count > 0:
            self.faiss_store.save_index()
            
            # Delete stored file
            doc_path = settings.DOCUMENTS_PATH / document_name
            if doc_path.exists():
                doc_path.unlink()
        
        return deleted_count
