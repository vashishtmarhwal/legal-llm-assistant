"""
API data models for request and response schemas.

This module defines Pydantic models for API requests and responses,
ensuring proper data validation and documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Enum for supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class SummaryType(str, Enum):
    """Enum for summary types."""
    BRIEF = "brief"  # 1-2 paragraphs
    COMPREHENSIVE = "comprehensive"  # Detailed summary
    EXECUTIVE = "executive"  # Executive summary
    LEGAL_POINTS = "legal_points"  # Bullet points of key legal implications


class ClauseType(str, Enum):
    """Enum for legal clause types."""
    JURISDICTION = "jurisdiction"
    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    INDEMNIFICATION = "indemnification"
    TERMINATION = "termination"
    GOVERNING_LAW = "governing_law"
    DISPUTE_RESOLUTION = "dispute_resolution"
    CONFIDENTIALITY = "confidentiality"
    PAYMENT_TERMS = "payment_terms"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    FORCE_MAJEURE = "force_majeure"
    WARRANTY = "warranty"
    NON_COMPETE = "non_compete"
    ALL = "all"  # Detect all clause types


class DocumentBase(BaseModel):
    """Base model for document information."""
    title: str = Field(..., description="Document title")
    description: Optional[str] = Field(None, description="Document description")
    document_type: DocumentType = Field(..., description="Type of document")


class DocumentCreate(DocumentBase):
    """Model for creating a document."""
    content: Optional[str] = Field(None, description="Text content if directly provided")
    # File handling is done via Form data, not in the JSON schema


class DocumentResponse(DocumentBase):
    """Model for document response."""
    id: str = Field(..., description="Document unique identifier")
    created_at: datetime = Field(..., description="Document creation timestamp")
    updated_at: datetime = Field(..., description="Document last update timestamp")
    page_count: Optional[int] = Field(None, description="Number of pages in the document")
    word_count: Optional[int] = Field(None, description="Word count in the document")
    is_processed: bool = Field(..., description="Whether the document has been processed")
    chunk_count: Optional[int] = Field(None, description="Number of chunks the document was split into")

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "id": "doc_123456",
                "title": "Commercial Lease Agreement",
                "description": "Commercial lease agreement for office space",
                "document_type": "pdf",
                "created_at": "2023-09-15T10:30:00Z",
                "updated_at": "2023-09-15T10:35:00Z",
                "page_count": 12,
                "word_count": 5240,
                "is_processed": True,
                "chunk_count": 23
            }
        }


class DocumentList(BaseModel):
    """Model for list of documents response."""
    total: int = Field(..., description="Total number of documents")
    documents: List[DocumentResponse] = Field(..., description="List of documents")


class TextChunk(BaseModel):
    """Model for a text chunk from a document."""
    id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Chunk content")
    page_number: Optional[int] = Field(None, description="Page number in the original document")
    chunk_index: int = Field(..., description="Index of the chunk in the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryRequest(BaseModel):
    """Model for document query request."""
    document_id: Optional[str] = Field(None, description="Document ID to query (if None, query all documents)")
    query: str = Field(..., description="Query text", min_length=3, max_length=500)
    top_k: int = Field(default=3, description="Number of results to return", ge=1, le=10)

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "document_id": "doc_123456",
                "query": "What are the termination conditions specified in the agreement?",
                "top_k": 3
            }
        }


class QueryResponse(BaseModel):
    """Model for document query response."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[TextChunk] = Field(..., description="Source chunks used for the answer")
    confidence_score: float = Field(..., description="Confidence score of the answer", ge=0, le=1)
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class SummarizeRequest(BaseModel):
    """Model for document summarization request."""
    document_id: str = Field(..., description="Document ID to summarize")
    summary_type: SummaryType = Field(default=SummaryType.COMPREHENSIVE, description="Type of summary to generate")
    max_length: Optional[int] = Field(None, description="Maximum length of summary in tokens")

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "document_id": "doc_123456",
                "summary_type": "executive",
                "max_length": 500
            }
        }


class SummarizeResponse(BaseModel):
    """Model for document summarization response."""
    document_id: str = Field(..., description="Document ID")
    summary_type: SummaryType = Field(..., description="Type of summary generated")
    summary: str = Field(..., description="Generated summary")
    word_count: int = Field(..., description="Word count of the summary")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class ClauseDetectionRequest(BaseModel):
    """Model for clause detection request."""
    document_id: str = Field(..., description="Document ID to analyze")
    clause_types: List[ClauseType] = Field(default=[ClauseType.ALL], description="Types of clauses to detect")

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "document_id": "doc_123456",
                "clause_types": ["termination", "governing_law", "dispute_resolution"]
            }
        }


class DetectedClause(BaseModel):
    """Model for a detected clause."""
    clause_type: ClauseType = Field(..., description="Type of the clause")
    content: str = Field(..., description="Content of the clause")
    page_number: Optional[int] = Field(None, description="Page number in the original document")
    confidence_score: float = Field(..., description="Detection confidence score", ge=0, le=1)
    chunk_id: str = Field(..., description="ID of the chunk containing this clause")


class ClauseDetectionResponse(BaseModel):
    """Model for clause detection response."""
    document_id: str = Field(..., description="Document ID")
    clauses: List[DetectedClause] = Field(..., description="Detected clauses")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class ErrorResponse(BaseModel):
    """Model for error responses."""
    status_code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Error message")
    detail: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Error details")


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    llm_status: str = Field(..., description="LLM service status")
    vector_store_status: str = Field(..., description="Vector store status")
    uptime_seconds: int = Field(..., description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Model for LLM model information."""
    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Model provider (e.g., 'huggingface')")
    context_window: int = Field(..., description="Context window size in tokens")
    embedding_dimensions: Optional[int] = Field(None, description="Embedding dimensions (for embedding models)")
    quantization: Optional[str] = Field(None, description="Quantization type if applicable")
    description: Optional[str] = Field(None, description="Additional model description")


class ModelsResponse(BaseModel):
    """Model for available models response."""
    llm_model: ModelInfo = Field(..., description="Current LLM model information")
    embedding_model: ModelInfo = Field(..., description="Current embedding model information")
    available_llm_models: List[str] = Field(..., description="List of available LLM models")
    available_embedding_models: List[str] = Field(..., description="List of available embedding models")