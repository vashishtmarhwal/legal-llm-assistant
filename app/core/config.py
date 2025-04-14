"""Configuration settings for the Legal Assistant application."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


class LLMConfig(BaseModel):
    """Configuration for LLM models."""

    model_id: str = Field(
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        description="The model ID to use for generating responses",
    )
    embedding_model_id: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="The model ID to use for embeddings",
    )
    max_new_tokens: int = Field(
        default=512, description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.1, description="Sampling temperature for generation"
    )
    top_p: float = Field(default=0.95, description="Top-p sampling parameter")
    quantization: Optional[str] = Field(
        default="4bit", description="Quantization level (4bit, 8bit, or None)"
    )


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    provider: str = Field(
        default="faiss", description="Vector store provider (faiss or weaviate)"
    )
    weaviate_url: Optional[str] = Field(
        default=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        description="Weaviate server URL",
    )
    collection_name: str = Field(
        default="legal_documents", description="Collection name for the vector store"
    )
    distance_metric: str = Field(
        default="cosine", description="Distance metric for vector similarity"
    )


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    chunk_size: int = Field(
        default=1000, description="Size of text chunks for processing"
    )
    chunk_overlap: int = Field(
        default=200, description="Overlap between consecutive chunks"
    )
    separator: str = Field(default="\n", description="Separator for chunking")


class APIConfig(BaseModel):
    """Configuration for the API server."""

    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    title: str = Field(default="Legal Assistant API", description="API title")
    description: str = Field(
        default="AI-powered legal assistant that extracts and summarizes legal documents",
        description="API description",
    )
    version: str = Field(default="0.1.0", description="API version")


class UIConfig(BaseModel):
    """Configuration for the UI."""

    theme: str = Field(default="default", description="Gradio theme")
    analytics_enabled: bool = Field(
        default=False, description="Enable Gradio analytics"
    )
    cache_examples: bool = Field(default=True, description="Cache examples")
    enable_queue: bool = Field(default=True, description="Enable request queue")


class AppConfig(BaseModel):
    """Main application configuration."""

    debug: bool = Field(
        default=os.getenv("DEBUG", "False").lower() == "true",
        description="Debug mode",
    )
    environment: str = Field(
        default=os.getenv("ENVIRONMENT", "development"),
        description="Application environment",
    )
    log_level: str = Field(
        default=os.getenv("LOG_LEVEL", "INFO"),
        description="Logging level",
    )
    allowed_file_types: List[str] = Field(
        default=["pdf", "docx", "txt"],
        description="Allowed file types for upload",
    )
    max_file_size_mb: int = Field(
        default=25, description="Maximum file size in MB"
    )
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    class Config:
        """Pydantic configuration."""

        case_sensitive = False


# Create app config instance
app_config = AppConfig()


def get_config() -> AppConfig:
    """Return the application configuration."""
    return app_config