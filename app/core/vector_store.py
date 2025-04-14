"""Vector store module for document embedding storage and retrieval."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS, Weaviate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
import weaviate
from weaviate.util import get_valid_uuid

from app.core.config import get_config

# Set up logging
logger = logging.getLogger(__name__)


class DocumentEmbeddingStore:
    """
    Manages document embeddings and vector storage.
    Supports both FAISS and Weaviate as backend vector stores.
    """
    
    def __init__(self, embedding_function):
        """
        Initialize the document embedding store.
        
        Args:
            embedding_function: Function to create embeddings
        """
        self.config = get_config().vector_store
        self.embedding_function = embedding_function
        self._vector_store = None
        
        logger.info(f"Initializing vector store with provider: {self.config.provider}")

    @property
    def vector_store(self) -> VectorStore:
        """
        Get the vector store.
        Initializes if not already done.
        
        Returns:
            VectorStore: The vector store instance
        """
        if self._vector_store is None:
            self._initialize_vector_store()
        return self._vector_store

    def _initialize_vector_store(self) -> None:
        """Initialize the vector store based on configuration."""
        provider = self.config.provider.lower()
        
        if provider == "weaviate":
            self._initialize_weaviate()
        elif provider == "faiss":
            self._initialize_faiss()
        else:
            logger.warning(f"Unknown vector store provider: {provider}, defaulting to FAISS")
            self._initialize_faiss()

    def _initialize_weaviate(self) -> None:
        """Initialize Weaviate vector store."""
        try:
            client = weaviate.Client(url=self.config.weaviate_url)
            
            # Check if the collection exists
            collection_name = self.config.collection_name
            
            # Check if collection exists
            if client.schema.exists(collection_name):
                logger.info(f"Using existing Weaviate collection: {collection_name}")
            else:
                # Create the collection
                class_obj = {
                    "class": collection_name,
                    "vectorizer": "none",  # We'll provide our own vectors
                    "properties": [
                        {
                            "name": "text",
                            "dataType": ["text"],
                        },
                        {
                            "name": "source",
                            "dataType": ["string"],
                        },
                        {
                            "name": "metadata",
                            "dataType": ["object"],
                        }
                    ],
                }
                client.schema.create_class(class_obj)
                logger.info(f"Created new Weaviate collection: {collection_name}")
                
            # Create the vector store
            self._vector_store = Weaviate(
                client=client,
                index_name=collection_name,
                text_key="text",
                embedding=self.embedding_function,
                by_text=False,
                attributes=["source", "metadata"],
            )
            
            logger.info("Weaviate vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Weaviate: {str(e)}")
            logger.info("Falling back to FAISS")
            self._initialize_faiss()

    def _initialize_faiss(self) -> None:
        """Initialize FAISS vector store."""
        try:
            # Create an empty FAISS index
            dimension = 384  # Default for all-MiniLM-L6-v2
            index = faiss.IndexFlatIP(dimension)
            
            # Create the vector store
            self._vector_store = FAISS(
                embedding_function=self.embedding_function,
                index=index,
                docstore={},
                index_to_docstore_id={},
            )
            
            logger.info("FAISS vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        try:
            if not documents:
                logger.warning("No documents to add to vector store")
                return
                
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.vector_store.add_documents(documents)
            logger.info("Documents added successfully")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform similarity search in the vector store.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_metadata: Optional metadata filtering
            
        Returns:
            List[Document]: List of similar documents
        """
        try:
            logger.info(f"Performing similarity search for query: {query[:50]}...")
            
            # Apply filtering if provided
            if filter_metadata and hasattr(self.vector_store, "similarity_search_with_filter_dict"):
                docs = self.vector_store.similarity_search_with_filter_dict(
                    query, filter_metadata, k=k
                )
            else:
                docs = self.vector_store.similarity_search(query, k=k)
                
            logger.info(f"Found {len(docs)} relevant documents")
            return docs
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []

    def save_local(self, file_path: Union[str, Path] = None) -> str:
        """
        Save the vector store locally (only for FAISS).
        
        Args:
            file_path: Optional path to save to
            
        Returns:
            str: Path where the vector store was saved
        """
        if self.config.provider.lower() != "faiss":
            logger.warning(f"Save local is only supported for FAISS, not {self.config.provider}")
            return None
            
        try:
            # Create temp directory if no path provided
            if file_path is None:
                temp_dir = tempfile.mkdtemp(prefix="legal_assistant_vectorstore_")
                file_path = os.path.join(temp_dir, "faiss_index")
                
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving FAISS index to {path}")
            self.vector_store.save_local(str(path))
            logger.info(f"FAISS index saved successfully to {path}")
            
            return str(path)
            
        except Exception as e:
            logger.error(f"Error saving vector store locally: {str(e)}")
            return None

    def load_local(self, file_path: Union[str, Path]) -> bool:
        """
        Load a vector store from local storage (only for FAISS).
        
        Args:
            file_path: Path to load from
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if self.config.provider.lower() != "faiss":
            logger.warning(f"Load local is only supported for FAISS, not {self.config.provider}")
            return False
            
        try:
            logger.info(f"Loading FAISS index from {file_path}")
            self._vector_store = FAISS.load_local(
                str(file_path),
                self.embedding_function
            )
            logger.info("FAISS index loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False

    def clear(self) -> bool:
        """
        Clear the vector store.
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        try:
            provider = self.config.provider.lower()
            
            if provider == "faiss":
                # Reinitialize FAISS index
                self._initialize_faiss()
                logger.info("FAISS vector store cleared successfully")
                return True
                
            elif provider == "weaviate":
                # Get client and delete objects
                client = self._vector_store._client
                collection_name = self.config.collection_name
                
                # Delete all objects in the collection
                client.batch.delete_objects(
                    collection_name,
                    where={"operator": "Equal", "path": ["id"], "valueString": "*"}
                )
                
                logger.info(f"Weaviate collection {collection_name} cleared successfully")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False