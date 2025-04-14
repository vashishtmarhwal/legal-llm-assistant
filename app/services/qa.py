"""Question answering service for legal documents."""

import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

from app.core.llm_manager import LLMManager
from app.services.extraction import ExtractionService

# Set up logging
logger = logging.getLogger(__name__)


class QAService:
    """
    Service for answering questions about legal documents.
    Provides context-aware responses using retrieval-augmented generation.
    """
    
    def __init__(self, extraction_service: Optional[ExtractionService] = None):
        """
        Initialize the QA service.
        
        Args:
            extraction_service: Optional extraction service to use
        """
        self.llm_manager = LLMManager()
        self.extraction_service = extraction_service or ExtractionService()
        self.qa_chain = self.llm_manager.create_qa_chain()
        
        logger.info("QA service initialized")

    async def answer_question(self, question: str, k: int = 4) -> Dict:
        """
        Answer a question based on the document context.
        
        Args:
            question: The question to answer
            k: Number of chunks to retrieve for context
            
        Returns:
            Dict: Answer results
        """
        try:
            logger.info(f"Answering question: {question}")
            
            # Retrieve relevant chunks
            relevant_chunks = self.extraction_service.retrieve_relevant_chunks(question, k)
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found to answer the question")
                return {
                    "status": "error",
                    "message": "No relevant information found to answer your question",
                    "answer": "I don't have enough information to answer this question based on the uploaded documents.",
                    "sources": []
                }
                
            # Create context from chunks
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            
            # Get sources for citation
            sources = [
                {
                    "content": chunk.page_content[:150] + "...",
                    "source": chunk.metadata.get("source", "Unknown"),
                    "page": chunk.metadata.get("page", None)
                }
                for chunk in relevant_chunks
            ]
            
            # Run the QA chain
            result = self.qa_chain(
                {"context": context, "question": question}
            )
            
            answer = result.get("output_text", "")
            
            logger.info("Question answered successfully")
            return {
                "status": "success",
                "message": "Question answered successfully",
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "status": "error",
                "message": f"Error answering question: {str(e)}",
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": []
            }

    async def ask_follow_up(self, question: str, previous_qa: List[Dict], k: int = 4) -> Dict:
        """
        Answer a follow-up question with context from previous Q&A.
        
        Args:
            question: The follow-up question
            previous_qa: List of previous questions and answers
            k: Number of chunks to retrieve for context
            
        Returns:
            Dict: Answer results
        """
        try:
            logger.info(f"Answering follow-up question: {question}")
            
            # Create context from previous Q&A
            qa_context = ""
            for qa in previous_qa[-3:]:  # Use last 3 Q&A pairs for context
                q = qa.get("question", "")
                a = qa.get("answer", "")
                if q and a:
                    qa_context += f"Q: {q}\nA: {a}\n\n"
                    
            # Create enhanced query with previous context
            enhanced_query = f"{qa_context}\nFollow-up question: {question}"
            
            # Retrieve relevant chunks with enhanced query
            relevant_chunks = self.extraction_service.retrieve_relevant_chunks(enhanced_query, k)
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found to answer the follow-up question")
                return {
                    "status": "error",
                    "message": "No relevant information found to answer your follow-up question",
                    "answer": "I don't have enough information to answer this follow-up question based on the uploaded documents.",
                    "sources": []
                }
                
            # Create document context from chunks
            doc_context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            
            # Combine both contexts
            full_context = f"{qa_context}\n\nRelevant Document Information:\n{doc_context}"
            
            # Get sources for citation
            sources = [
                {
                    "content": chunk.page_content[:150] + "...",
                    "source": chunk.metadata.get("source", "Unknown"),
                    "page": chunk.metadata.get("page", None)
                }
                for chunk in relevant_chunks
            ]
            
            # Create a custom prompt for follow-up
            follow_up_prompt = """
            You are answering a follow-up question about legal documents. 
            Use the conversation history and document context to provide an accurate answer.
            
            Previous conversation:
            {qa_context}
            
            Document context:
            {doc_context}
            
            Follow-up question: {question}
            
            Answer:
            """
            
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            # Create and run the chain
            prompt = PromptTemplate(
                template=follow_up_prompt,
                input_variables=["qa_context", "doc_context", "question"]
            )
            chain = LLMChain(llm=self.llm_manager.llm, prompt=prompt)
            
            result = chain.run(
                qa_context=qa_context,
                doc_context=doc_context,
                question=question
            )
            
            logger.info("Follow-up question answered successfully")
            return {
                "status": "success",
                "message": "Follow-up question answered successfully",
                "answer": result,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error answering follow-up question: {str(e)}")
            return {
                "status": "error",
                "message": f"Error answering follow-up question: {str(e)}",
                "answer": f"An error occurred while processing your follow-up question: {str(e)}",
                "sources": []
            }

    async def analyze_legal_issue(self, issue: str) -> Dict:
        """
        Analyze a specific legal issue based on documents.
        
        Args:
            issue: The legal issue to analyze
            
        Returns:
            Dict: Analysis results
        """
        try:
            logger.info(f"Analyzing legal issue: {issue}")
            
            # Create an enhanced query for legal issue analysis
            query = f"information related to the legal issue: {issue}"
            
            # Retrieve more chunks for comprehensive analysis
            relevant_chunks = self.extraction_service.retrieve_relevant_chunks(query, k=8)
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found for legal issue analysis")
                return {
                    "status": "error",
                    "message": "No relevant information found to analyze this legal issue",
                    "analysis": "I don't have enough information to analyze this legal issue based on the uploaded documents.",
                    "sources": []
                }
                
            # Create context from chunks
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            
            # Get sources for citation
            sources = [
                {
                    "content": chunk.page_content[:150] + "...",
                    "source": chunk.metadata.get("source", "Unknown"),
                    "page": chunk.metadata.get("page", None)
                }
                for chunk in relevant_chunks
            ]
            
            # Create a custom prompt for legal issue analysis
            analysis_prompt = """
            You are a legal assistant analyzing a specific legal issue based on provided documents.
            Provide a comprehensive analysis of the legal issue, considering the relevant document context.
            Include:
            1. Key legal considerations
            2. Potential risks or conflicts
            3. Relevant clauses or provisions
            4. Possible interpretations or precedents
            5. Recommended next steps or actions
            
            Legal issue to analyze: {issue}
            
            Document context:
            {context}
            
            Legal Analysis:
            """
            
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            # Create and run the chain
            prompt = PromptTemplate(
                template=analysis_prompt,
                input_variables=["issue", "context"]
            )
            chain = LLMChain(llm=self.llm_manager.llm, prompt=prompt)
            
            result = chain.run(
                issue=issue,
                context=context
            )
            
            logger.info("Legal issue analyzed successfully")
            return {
                "status": "success",
                "message": "Legal issue analyzed successfully",
                "analysis": result,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error analyzing legal issue: {str(e)}")
            return {
                "status": "error",
                "message": f"Error analyzing legal issue: {str(e)}",
                "analysis": f"An error occurred while analyzing the legal issue: {str(e)}",
                "sources": []
            }