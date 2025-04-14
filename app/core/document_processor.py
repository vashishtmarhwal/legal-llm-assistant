"""Document processing module for the Legal Assistant application."""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_core.documents import Document

from app.core.config import get_config


class DocumentProcessor:
    """Handles document loading, processing, and chunking."""

    def __init__(self):
        """Initialize the document processor with configuration."""
        config = get_config()
        self.chunking_config = config.chunking
        self.allowed_file_types = config.allowed_file_types
        self.max_file_size_mb = config.max_file_size_mb
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunking_config.chunk_size,
            chunk_overlap=self.chunking_config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False,
        )

    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate the file type and size.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if the file is valid, False otherwise
        """
        path = Path(file_path)
        
        # Check file extension
        extension = path.suffix.lower().lstrip(".")
        if extension not in self.allowed_file_types:
            return False
            
        # Check file size
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            return False
            
        return True

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict:
        """
        Extract metadata from the document.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict: Document metadata
        """
        path = Path(file_path)
        extension = path.suffix.lower().lstrip(".")
        metadata = {
            "source": str(path),
            "filename": path.name,
            "extension": extension,
            "file_size": os.path.getsize(path),
        }
        
        # Extract PDF-specific metadata if applicable
        if extension == "pdf":
            try:
                with open(path, "rb") as f:
                    pdf = pypdf.PdfReader(f)
                    if pdf.metadata:
                        for key, value in pdf.metadata.items():
                            clean_key = key.strip("/").lower()
                            metadata[f"pdf_{clean_key}"] = value
                    metadata["page_count"] = len(pdf.pages)
            except Exception as e:
                metadata["extraction_error"] = str(e)
                
        return metadata

    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a document from the given file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List[Document]: List of document objects
        """
        path = Path(file_path)
        extension = path.suffix.lower().lstrip(".")
        
        try:
            # Select the appropriate loader based on file extension
            if extension == "pdf":
                loader = PyPDFLoader(str(path))
            elif extension == "txt":
                loader = TextLoader(str(path))
            else:
                # Fall back to unstructured for other file types
                loader = UnstructuredFileLoader(str(path))
                
            documents = loader.load()
            
            # Add extra metadata
            metadata = self.extract_metadata(file_path)
            for doc in documents:
                doc.metadata.update(metadata)
                
            return documents
        except Exception as e:
            # Return a document with the error information
            return [Document(
                page_content="Error loading document: " + str(e),
                metadata={"source": str(path), "error": str(e)}
            )]

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for processing.
        
        Args:
            documents: List of document objects
            
        Returns:
            List[Document]: List of chunked document objects
        """
        return self.text_splitter.split_documents(documents)

    def process_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Process a document by loading and chunking it.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List[Document]: List of processed document chunks
        """
        # Validate the file first
        if not self.validate_file(file_path):
            return [Document(
                page_content="Invalid file. Please ensure the file is a PDF, DOCX, or TXT "
                "and is under the maximum size limit.",
                metadata={"source": str(file_path), "error": "Invalid file"}
            )]
            
        # Load and chunk the document
        documents = self.load_document(file_path)
        chunked_docs = self.chunk_documents(documents)
        
        return chunked_docs

    async def process_uploaded_file(self, file) -> List[Document]:
        """
        Process an uploaded file by saving it temporarily and processing it.
        
        Args:
            file: File object from FastAPI
            
        Returns:
            List[Document]: List of processed document chunks
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            # Write uploaded file content to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
            
        try:
            # Process the temporary file
            result = self.process_document(temp_path)
            return result
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)