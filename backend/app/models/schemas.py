
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SourceChunk(BaseModel):
    """Represents a source text chunk used in the RAG response."""
    
    text: str = Field(..., description="The actual text content of the chunk")
    document_name: str = Field(..., description="Name of the source document")
    chunk_index: int = Field(..., description="Index of this chunk in the document")
    similarity_score: float = Field(..., description="Similarity score from vector search")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "The Constitution of Nepal guarantees fundamental rights...",
                "document_name": "Constitution-of-Nepal-2072.pdf",
                "chunk_index": 15,
                "similarity_score": 0.89
            }
        }


class UploadResponse(BaseModel):
    """Response model for document upload endpoint."""
    
    success: bool = Field(..., description="Whether the upload was successful")
    message: str = Field(..., description="Status message")
    document_name: str = Field(..., description="Name of the uploaded document")
    total_chunks: int = Field(..., description="Number of chunks created from the document")
    processing_time: float = Field(..., description="Time taken to process in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Document processed and indexed successfully",
                "document_name": "legal_document.pdf",
                "total_chunks": 45,
                "processing_time": 2.34
            }
        }


class AskRequest(BaseModel):
    """Request model for the ask/question endpoint."""
    
    question: str = Field(
        ..., 
        min_length=3,
        max_length=1000,
        description="The question to ask about the documents"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the fundamental rights in Nepal's constitution?",
                "top_k": 5
            }
        }


class AskResponse(BaseModel):
    """Response model for the ask/question endpoint."""
    
    answer: str = Field(..., description="The generated answer from the LLM")
    sources: List[SourceChunk] = Field(
        default_factory=list,
        description="List of source chunks used to generate the answer"
    )
    question: str = Field(..., description="The original question asked")
    processing_time: float = Field(..., description="Total processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The fundamental rights in Nepal's constitution include...",
                "sources": [],
                "question": "What are the fundamental rights?",
                "processing_time": 1.23
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="API version")
    index_loaded: bool = Field(..., description="Whether FAISS index is loaded")
    total_documents: int = Field(..., description="Number of indexed documents")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
