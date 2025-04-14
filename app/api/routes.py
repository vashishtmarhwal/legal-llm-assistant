"""
API routes for the legal assistant application.

This module defines FastAPI routes for document processing, querying,
summarization, and clause detection.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.models import (
    ClauseDetectionRequest,
    ClauseDetectionResponse,
    DocumentCreate,
    DocumentList,
    DocumentResponse,
    DocumentType,
    HealthResponse,
    ModelsResponse,
    QueryRequest,
    QueryResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.core.config import settings
from app.core.document_processor import DocumentProcessor
from app.core.llm_manager import get_llm_instance
from app.services.clause_detection import detect_clauses
from app.services.extraction import extract_text
from app.services.qa import answer_query
from app.services.summarization import generate_summary

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_document_processor(request: Request) -> DocumentProcessor:
    """
    Dependency to get the document processor instance.
    
    Args:
        request: FastAPI request object
        
    Returns:
        DocumentProcessor: Document processor instance
    """
    if not hasattr(request.app.state, "document_processor"):
        request.app.state.document_processor = DocumentProcessor(
            vector_store=request.app.state.vector_store,
            llm=request.app.state.llm
        )
    return request.app.state.document_processor


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    """
    Check the health status of the API and its dependencies.
    
    Args:
        request: FastAPI request object
        
    Returns:
        HealthResponse: Health status information
    """
    start_time = getattr(request.app.state, "start_time", datetime.now())
    uptime = int((datetime.now() - start_time).total_seconds())
    
    # Check LLM status
    llm_status = "healthy"
    try:
        llm = request.app.state.llm
        if llm is None:
            llm_status = "not_initialized"
    except Exception as e:
        logger.error(f"Error checking LLM status: {str(e)}")
        llm_status = "error"
    
    # Check vector store status
    vector_store_status = "healthy"
    try:
        vector_store = request.app.state.vector_store
        if vector_store is None:
            vector_store_status = "not_initialized"
        # Quick validation check
        if hasattr(vector_store, "is_ready") and not vector_store.is_ready():
            vector_store_status = "not_ready"
    except Exception as e:
        logger.error(f"Error checking vector store status: {str(e)}")
        vector_store_status = "error"
    
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        llm_status=llm_status,
        vector_store_status=vector_store_status,
        uptime_seconds=uptime
    )


@router.get("/models", response_model=ModelsResponse, tags=["System"])
async def get_models_info(request: Request) -> ModelsResponse:
    """
    Get information about the currently loaded models and available models.
    
    Args:
        request: FastAPI request object
        
    Returns:
        ModelsResponse: Information about models
    """
    # Current models
    llm = request.app.state.llm
    vector_store = request.app.state.vector_store
    
    # Get LLM info
    llm_model_info = {
        "name": getattr(llm, "model_name", settings.LLM_MODEL),
        "provider": "huggingface",
        "context_window": getattr(llm, "context_window", 4096),
        "quantization": getattr(llm, "quantization", "8bit"),
        "description": "Open-source LLM from Hugging Face"
    }
    
    # Get embedding model info
    embedding_model_info = {
        "name": getattr(vector_store, "embedding_model_name", settings.EMBEDDING_MODEL),
        "provider": "huggingface",
        "embedding_dimensions": getattr(vector_store, "embedding_dimensions", 384),
        "description": "Sentence transformer embedding model"
    }
    
    # Available models - in a real app, this would be dynamically generated
    # For this example, we'll hardcode some popular open-source options
    available_llm_models = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-Instruct-v0.1", 
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "google/gemma-7b-it",
        "NousResearch/Nous-Hermes-2-Mistral-7B",
        "stabilityai/stablelm-3b-4e1t"
    ]
    
    available_embedding_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "intfloat/e5-base-v2",
        "BAAI/bge-small-en",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    ]
    
    return ModelsResponse(
        llm_model=llm_model_info,
        embedding_model=embedding_model_info,
        available_llm_models=available_llm_models,
        available_embedding_models=available_embedding_models
    )


@router.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def upload_document(
    file: Optional[UploadFile] = File(None),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    document_type: DocumentType = Form(...),
    content: Optional[str] = Form(None),
    document_processor: DocumentProcessor = Depends(get_document_processor)
) -> DocumentResponse:
    """
    Upload and process a new document.
    
    Args:
        file: Document file to upload
        title: Document title
        description: Document description
        document_type: Type of document
        content: Raw text content (alternative to file upload)
        document_processor: Document processor dependency
        
    Returns:
        DocumentResponse: Created document information
    """
    # Validate that either file or content is provided
    if file is None and content is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either file or content must be provided"
        )
    
    # Check file size if a file is provided
    if file is not None:
        file_size_mb = 0
        try:
            file.file.seek(0, 2)  # Move to end of file
            file_size_mb = file.file.tell() / (1024 * 1024)
            file.file.seek(0)  # Reset file pointer
        except Exception as e:
            logger.error(f"Error checking file size: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file"
            )
        
        if file_size_mb > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds the maximum allowed size of {settings.MAX_UPLOAD_SIZE}MB"
            )
    
    # Generate document ID
    document_id = f"doc_{uuid4().hex[:10]}"
    created_at = datetime.now()
    
    try:
        # Extract text from the document
        document_text = ""
        page_count = None
        word_count = None
        
        if file is not None:
            # Save file to a temporary location
            with NamedTemporaryFile(delete=False, suffix=f".{document_type.value}") as temp_file:
                try:
                    temp_file.write(await file.read())
                    temp_file_path = Path(temp_file.name)
                    
                    # Extract text from the file
                    extraction_result = await extract_text(
                        file_path=temp_file_path,
                        document_type=document_type
                    )
                    document_text = extraction_result["text"]
                    page_count = extraction_result.get("page_count")
                    word_count = len(document_text.split())
                finally:
                    # Clean up the temporary file
                    try:
                        Path(temp_file.name).unlink(missing_ok=True)
                    except Exception as e:
                        logger.error(f"Error removing temporary file: {str(e)}")
        else:
            # Use provided text content
            document_text = content
            word_count = len(document_text.split())
        
        # Process and store the document
        chunk_count = await document_processor.process_document(
            document_id=document_id,
            text=document_text,
            metadata={
                "title": title,
                "description": description,
                "document_type": document_type.value,
                "created_at": created_at.isoformat(),
            }
        )
        
        # Create the response
        return DocumentResponse(
            id=document_id,
            title=title,
            description=description,
            document_type=document_type,
            created_at=created_at,
            updated_at=created_at,
            page_count=page_count,
            word_count=word_count,
            is_processed=True,
            chunk_count=chunk_count
        )
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.get("/documents", response_model=DocumentList, tags=["Documents"])
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    document_processor: DocumentProcessor = Depends(get_document_processor)
) -> DocumentList:
    """
    List all documents.
    
    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        document_processor: Document processor dependency
        
    Returns:
        DocumentList: List of documents
    """
    try:
        documents = await document_processor.list_documents(skip=skip, limit=limit)
        total = await document_processor.count_documents()
        
        return DocumentList(
            total=total,
            documents=documents
        )
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@router.get("/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document(
    document_id: str,
    document_processor: DocumentProcessor = Depends(get_document_processor)
) -> DocumentResponse:
    """
    Get a document by ID.
    
    Args:
        document_id: Document ID
        document_processor: Document processor dependency
        
    Returns:
        DocumentResponse: Document information
    """
    try:
        document = await document_processor.get_document(document_id)
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found"
            )
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting document: {str(e)}"
        )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Documents"])
async def delete_document(
    document_id: str,
    document_processor: DocumentProcessor = Depends(get_document_processor)
) -> None:
    """
    Delete a document by ID.
    
    Args:
        document_id: Document ID
        document_processor: Document processor dependency
    """
    try:
        deleted = await document_processor.delete_document(document_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )
    return JSONResponse(status_code=204, content={})


@router.post("/query", response_model=QueryResponse, tags=["Querying"])
async def query_documents(
    query_request: QueryRequest,
    document_processor: DocumentProcessor = Depends(get_document_processor)
) -> QueryResponse:
    """
    Query documents and get answers.
    
    Args:
        query_request: Query request
        document_processor: Document processor dependency
        
    Returns:
        QueryResponse: Query response with answer and sources
    """
    start_time = time.time()
    
    try:
        # Get relevant document chunks
        relevant_chunks = await document_processor.search_chunks(
            query=query_request.query,
            document_id=query_request.document_id,
            top_k=query_request.top_k
        )
        
        if not relevant_chunks:
            return QueryResponse(
                query=query_request.query,
                answer="I couldn't find any relevant information to answer your question in the provided documents.",
                sources=[],
                confidence_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Generate answer
        answer_result = await answer_query(
            query=query_request.query,
            chunks=relevant_chunks,
            llm=get_llm_instance()
        )
        
        return QueryResponse(
            query=query_request.query,
            answer=answer_result["answer"],
            sources=relevant_chunks,
            confidence_score=answer_result.get("confidence_score", 0.85),
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
    
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying documents: {str(e)}"
        )


@router.post("/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize_document(
    summarize_request: SummarizeRequest,
    document_processor: DocumentProcessor = Depends(get_document_processor)
) -> SummarizeResponse:
    """
    Generate a summary of a document.
    
    Args:
        summarize_request: Summarization request
        document_processor: Document processor dependency
        
    Returns:
        SummarizeResponse: Generated summary
    """
    start_time = time.time()
    
    try:
        # Get document chunks
        document_chunks = await document_processor.get_document_chunks(
            document_id=summarize_request.document_id
        )
        
        if not document_chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {summarize_request.document_id} not found or has no content"
            )
        
        # Generate summary
        summary_result = await generate_summary(
            chunks=document_chunks,
            summary_type=summarize_request.summary_type,
            max_length=summarize_request.max_length,
            llm=get_llm_instance()
        )
        
        return SummarizeResponse(
            document_id=summarize_request.document_id,
            summary_type=summarize_request.summary_type,
            summary=summary_result["summary"],
            word_count=len(summary_result["summary"].split()),
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error summarizing document: {str(e)}"
        )


@router.post("/detect-clauses", response_model=ClauseDetectionResponse, tags=["Clause Detection"])
async def detect_document_clauses(
    detection_request: ClauseDetectionRequest,
    document_processor: DocumentProcessor = Depends(get_document_processor)
) -> ClauseDetectionResponse:
    """
    Detect legal clauses in a document.
    
    Args:
        detection_request: Clause detection request
        document_processor: Document processor dependency
        
    Returns:
        ClauseDetectionResponse: Detected clauses
    """
    start_time = time.time()
    
    try:
        # Get document chunks
        document_chunks = await document_processor.get_document_chunks(
            document_id=detection_request.document_id
        )
        
        if not document_chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {detection_request.document_id} not found or has no content"
            )
        
        # Detect clauses
        clauses_result = await detect_clauses(
            chunks=document_chunks,
            clause_types=detection_request.clause_types,
            llm=get_llm_instance()
        )
        
        return ClauseDetectionResponse(
            document_id=detection_request.document_id,
            clauses=clauses_result["clauses"],
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting clauses: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting clauses: {str(e)}"
        )