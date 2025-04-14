"""Summarization service for legal documents."""

import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document
from nltk.tokenize import sent_tokenize
import nltk

from app.core.llm_manager import LLMManager
from app.services.extraction import ExtractionService

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Set up logging
logger = logging.getLogger(__name__)


class SummarizationService:
    """
    Service for summarizing legal documents.
    Provides document summarization at different levels of granularity.
    """
    
    def __init__(self, extraction_service: Optional[ExtractionService] = None):
        """
        Initialize the summarization service.
        
        Args:
            extraction_service: Optional extraction service to use
        """
        self.llm_manager = LLMManager()
        self.extraction_service = extraction_service or ExtractionService()
        self.summarization_chain = self.llm_manager.create_summarization_chain()
        
        logger.info("Summarization service initialized")

    def _chunk_text_by_sentences(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Chunk text by sentences to avoid exceeding token limits.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum chunk size in characters
            
        Returns:
            List[str]: List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed the max chunk size, start a new chunk
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    async def summarize_document(self, query: str = "summarize this document") -> Dict:
        """
        Summarize an entire document.
        
        Args:
            query: Optional query to guide the summarization
            
        Returns:
            Dict: Summary results
        """
        try:
            logger.info(f"Summarizing document with query: {query}")
            
            # Retrieve relevant chunks
            chunks = self.extraction_service.retrieve_relevant_chunks(query, k=10)
            
            if not chunks:
                logger.warning("No relevant chunks found for summarization")
                return {
                    "status": "error",
                    "message": "No document content found to summarize",
                    "summary": ""
                }
                
            # Combine chunks into a single text
            combined_text = "\n\n".join([chunk.page_content for chunk in chunks])
            
            # Check if text is too long for a single summary
            if len(combined_text) > 4000:
                logger.info("Document is large, performing hierarchical summarization")
                return await self.hierarchical_summarize(chunks)
            
            # Generate summary
            result = self.summarization_chain.run(text=combined_text)
            
            logger.info("Document summarized successfully")
            return {
                "status": "success",
                "message": "Document summarized successfully",
                "summary": result,
                "chunk_count": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error summarizing document: {str(e)}")
            return {
                "status": "error",
                "message": f"Error summarizing document: {str(e)}",
                "summary": ""
            }

    async def summarize_chunk(self, chunk: Document) -> str:
        """
        Summarize a single document chunk.
        
        Args:
            chunk: Document chunk to summarize
            
        Returns:
            str: Summary text
        """
        try:
            logger.info(f"Summarizing chunk: {chunk.page_content[:50]}...")
            
            # Generate summary
            result = self.summarization_chain.run(text=chunk.page_content)
            
            return result
            
        except Exception as e:
            logger.error(f"Error summarizing chunk: {str(e)}")
            return f"Error summarizing content: {str(e)}"

    async def hierarchical_summarize(self, chunks: List[Document]) -> Dict:
        """
        Perform hierarchical summarization for large documents.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dict: Summary results
        """
        try:
            logger.info(f"Performing hierarchical summarization with {len(chunks)} chunks")
            
            # First level: summarize each chunk
            chunk_summaries = []
            for chunk in chunks:
                summary = await self.summarize_chunk(chunk)
                chunk_summaries.append(summary)
                
            # Second level: summarize the summaries
            combined_summaries = "\n\n".join(chunk_summaries)
            
            # If the combined summaries are still too long, chunk them
            if len(combined_summaries) > 4000:
                logger.info("Combined summaries still too long, chunking again")
                summary_chunks = self._chunk_text_by_sentences(combined_summaries)
                final_summaries = []
                
                for s_chunk in summary_chunks:
                    result = self.summarization_chain.run(text=s_chunk)
                    final_summaries.append(result)
                    
                # Final summary of summaries
                final_text = "\n\n".join(final_summaries)
                final_summary = self.summarization_chain.run(text=final_text)
            else:
                # Final summary
                final_summary = self.summarization_chain.run(text=combined_summaries)
                
            logger.info("Hierarchical summarization completed successfully")
            return {
                "status": "success",
                "message": "Document summarized successfully using hierarchical summarization",
                "summary": final_summary,
                "chunk_count": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error during hierarchical summarization: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during hierarchical summarization: {str(e)}",
                "summary": ""
            }

    async def executive_summary(self) -> Dict:
        """
        Generate an executive summary of the document.
        
        Returns:
            Dict: Executive summary results
        """
        try:
            logger.info("Generating executive summary")
            
            # Custom query for executive summary
            query = "key points of this legal document for executives"
            
            # Retrieve relevant chunks
            chunks = self.extraction_service.retrieve_relevant_chunks(query, k=5)
            
            if not chunks:
                logger.warning("No relevant chunks found for executive summary")
                return {
                    "status": "error",
                    "message": "No document content found to summarize",
                    "summary": ""
                }
                
            # Create executive summary prompt
            executive_prompt = """
            Create a concise executive summary of this legal document. 
            Focus only on the most critical business implications, key risks, major obligations, 
            and strategic considerations. The summary should be brief, direct, and focused on 
            what executives would consider most important.
            
            Text to summarize:
            {text}
            
            Executive Summary:
            """
            
            # Create a custom chain for executive summary
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            prompt = PromptTemplate(template=executive_prompt, input_variables=["text"])
            exec_chain = LLMChain(llm=self.llm_manager.llm, prompt=prompt)
            
            # Combine chunks into a single text
            combined_text = "\n\n".join([chunk.page_content for chunk in chunks])
            
            # Generate executive summary
            result = exec_chain.run(text=combined_text)
            
            logger.info("Executive summary generated successfully")
            return {
                "status": "success",
                "message": "Executive summary generated successfully",
                "summary": result,
                "chunk_count": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating executive summary: {str(e)}",
                "summary": ""
            }