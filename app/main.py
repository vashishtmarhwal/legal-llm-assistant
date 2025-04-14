"""
Legal Assistant - Main Application Entry Point

This module initializes and runs the FastAPI application with Gradio UI integration.
It sets up necessary components, logging, and serves as the entry point for the application.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.llm_manager import get_llm_instance
from app.core.vector_store import VectorStore
from app.ui.interface import create_interface


# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """
    Initialize application resources and cleanup on shutdown.
    
    Args:
        app: FastAPI application instance
    """
    # Initialize resources
    logger.info("Initializing application resources...")
    
    # Initialize LLM
    app.state.llm = get_llm_instance()
    logger.info(f"LLM initialized: {type(app.state.llm).__name__}")
    
    # Initialize vector store
    app.state.vector_store = VectorStore(
        embedding_model=settings.EMBEDDING_MODEL,
        vector_db_path=settings.VECTOR_DB_PATH
    )
    logger.info(f"Vector store initialized at: {settings.VECTOR_DB_PATH}")
    
    yield
    
    # Cleanup resources
    logger.info("Cleaning up resources...")
    if hasattr(app.state, "vector_store"):
        await app.state.vector_store.close()
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        lifespan=lifespan,
        docs_url=settings.DOCS_URL,
        redoc_url=settings.REDOC_URL,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router
    app.include_router(api_router, prefix=settings.API_PREFIX)
    
    # Mount static files if needed
    if settings.SERVE_STATIC_FILES:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    
    return app


def create_gradio_app() -> gr.Blocks:
    """
    Create Gradio interface for the application.
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    return create_interface()


app = create_app()
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )