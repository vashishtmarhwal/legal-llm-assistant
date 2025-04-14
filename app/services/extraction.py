"""Extraction service for processing legal documents."""

import logging
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document

from app.core.document_processor import DocumentProcessor
from app.core.llm_manager import LLMManager
from app.core.vector_store import DocumentEmbeddingStore

# Set up logging
logger = logging.getLogger(__name__)


class ExtractionService:
    """
    Service for extracting and processing legal documents.
    Handles document loading, chunking, and embedding.
    """
    
    def __init__(self):
        """Initialize the extraction service."""
        self.document_processor = DocumentProcessor()
        self.llm_manager = LLMManager()
        
        # Create embedding function
        embedding_function = lambda texts: self.llm_manager.get_embeddings(texts)
        
        # Initialize vector store
        self.vector_store = DocumentEmbeddingStore(embedding_function)
        
        logger.info("Extraction service initialized")

    async def process_file(self, file) -> Dict:
        """
        Process an uploaded file.
        
        Args:
            file: File object from FastAPI
            
        Returns:
            Dict: Processing results
        """
        try:
            logger.info(f"Processing file: {file.filename}")
            
            # Process the document
            documents = await self.document_processor.process_uploaded_file(file)
            
            if not documents:
                logger.warning("No documents extracted from file")
                return {
                    "status": "error",
                    "message": "No content could be extracted from the file",
                    "document_count": 0
                }
                
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            # Get summary stats
            doc_count = len(documents)
            total_tokens = sum(len(doc.page_content.split()) for doc in documents)
            
            logger.info(f"File processed successfully: {doc_count} chunks, {total_tokens} tokens")
            
            return {
                "status": "success",
                "message": f"File processed successfully",
                "document_count": doc_count,
                "total_tokens": total_tokens,
                "metadata": documents[0].metadata if documents else {}
            }
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing file: {str(e)}",
                "document_count": 0
            }

    def retrieve_relevant_chunks(
        self, 
        query: str, 
        k: int = 4,
        filter_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Retrieve chunks relevant to a query.
        
        Args:
            query: Query text
            k: Number of chunks to retrieve
            filter_metadata: Optional metadata filtering
            
        Returns:
            List[Document]: List of relevant document chunks
        """
        logger.info(f"Retrieving {k} chunks relevant to query: {query[:50]}...")
        return self.vector_store.similarity_search(query, k, filter_metadata)

    def clear_documents(self) -> Dict:
        """
        Clear all documents from the vector store.
        
        Returns:
            Dict: Operation results
        """
        try:
            result = self.vector_store.clear()
            
            if result:
                logger.info("Vector store cleared successfully")
                return {
                    "status": "success",
                    "message": "All documents have been removed"
                }
            else:
                logger.warning("Failed to clear vector store")
                return {
                    "status": "error",
                    "message": "Failed to clear documents"
                }
                
        except Exception as e:
            logger.error(f"Error clearing documents: {str(e)}")
            return {
                "status": "error",
                "message": f"Error clearing documents: {str(e)}"
            }

    def save_vector_store(self) -> Dict:
        """
        Save the vector store locally.
        
        Returns:
            Dict: Operation results with save path
        """
        try:
            save_path = self.vector_store.save_local()
            
            if save_path:
                logger.info(f"Vector store saved to {save_path}")
                return {
                    "status": "success",
                    "message": "Vector store saved successfully",
                    "path": save_path
                }
            else:
                logger.warning("Failed to save vector store")
                return {
                    "status": "error",
                    "message": "Failed to save vector store"
                }
                
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return {
                "status": "error",
                "message": f"Error saving vector store: {str(e)}"
            }

    def load_vector_store(self, file_path: str) -> Dict:
        """
        Load a vector store from local storage.
        
        Args:
            file_path: Path to the vector store
            
        Returns:
            Dict: Operation results
        """
        try:
            result = self.vector_store.load_local(file_path)
            
            if result:
                logger.info(f"Vector store loaded from {file_path}")
                return {
                    "status": "success",
                    "message": "Vector store loaded successfully"
                }
            else:
                logger.warning(f"Failed to load vector store from {file_path}")
                return {
                    "status": "error",
                    "message": "Failed to load vector store"
                }
                
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return {
                "status": "error",
                "message": f"Error loading vector store: {str(e)}"
            }