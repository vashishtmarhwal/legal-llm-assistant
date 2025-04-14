"""LLM manager module for handling language model operations."""

import logging
import os
from typing import Dict, List, Optional, Union

import torch
from huggingface_hub import login
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.language_models import BaseLLM
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Pipeline,
    TextStreamer,
    pipeline,
)

from app.core.config import get_config

# Set up logging
logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manages language models for the application.
    Handles model loading, optimization, and provides various LLM functionalities.
    """
    
    def __init__(self):
        """Initialize the LLM manager with configuration."""
        self.config = get_config().llm
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Log the configuration
        logger.info(f"Initializing LLM Manager with device: {self.device}")
        logger.info(f"Using model: {self.config.model_id}")
        logger.info(f"Using embedding model: {self.config.embedding_model_id}")
        
        # Initialize models as None
        self._llm = None
        self._embedding_model = None
        
        # Try to load HF token
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            login(token=hf_token)

    def _load_quantized_model(self) -> AutoModelForCausalLM:
        """
        Load a quantized model based on configuration.
        
        Returns:
            AutoModelForCausalLM: The loaded model
        """
        quantization = self.config.quantization
        model_id = self.config.model_id
        
        # Configure quantization settings
        if quantization == "4bit":
            logger.info("Loading model with 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            logger.info("Loading model with 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            logger.info("Loading model without quantization")
            quantization_config = None
            
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        
        return model

    def _create_pipeline(self) -> Pipeline:
        """
        Create a text generation pipeline.
        
        Returns:
            Pipeline: The text generation pipeline
        """
        model_id = self.config.model_id
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Load model
        if self.config.quantization and self.device == "cuda":
            model = self._load_quantized_model()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            )
            
        # Create text streamer for streaming output
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Create pipeline
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            streamer=streamer,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        return text_pipeline

    @property
    def llm(self) -> BaseLLM:
        """
        Get the language model.
        Loads the model if not loaded already.
        
        Returns:
            BaseLLM: The language model
        """
        if self._llm is None:
            logger.info("Loading LLM model...")
            
            # Create LLM pipeline
            pipe = self._create_pipeline()
            
            # Initialize streaming callbacks
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            # Create LangChain LLM from pipeline
            self._llm = HuggingFacePipeline(
                pipeline=pipe,
                callback_manager=callback_manager,
                model_id=self.config.model_id,
            )
            
            logger.info("LLM loaded successfully")
            
        return self._llm

    @property
    def embedding_model(self) -> SentenceTransformer:
        """
        Get the embedding model.
        Loads the model if not loaded already.
        
        Returns:
            SentenceTransformer: The embedding model
        """
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.config.embedding_model_id}")
            
            self._embedding_model = SentenceTransformer(
                self.config.embedding_model_id,
                device=self.device
            )
            
            logger.info("Embedding model loaded successfully")
            
        return self._embedding_model

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embeddings
        """
        return self.embedding_model.encode(texts).tolist()

    def create_qa_chain(self) -> load_qa_chain:
        """
        Create a question answering chain.
        
        Returns:
            Chain: Question answering chain
        """
        # Define the QA prompt template
        prompt_template = """
        You are a legal assistant AI specialized in analyzing legal documents. 
        You've been given context from a legal document. Use this context to answer the question.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create and return the chain
        return load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=prompt,
        )

    def create_summarization_chain(self) -> LLMChain:
        """
        Create a summarization chain.
        
        Returns:
            LLMChain: Summarization chain
        """
        # Define the summarization prompt template
        prompt_template = """
        You are a legal assistant AI specialized in analyzing legal documents. 
        Summarize the following legal text in a clear, concise, and professional manner. 
        Focus on extracting the key points, obligations, rights, and important clauses.
        
        Text to summarize:
        {text}
        
        Summary:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"]
        )
        
        # Create and return the chain
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
        )

    def create_clause_detection_chain(self) -> LLMChain:
        """
        Create a clause detection chain.
        
        Returns:
            LLMChain: Clause detection chain
        """
        # Define the clause detection prompt template
        prompt_template = """
        You are a legal assistant AI specialized in detecting important clauses in legal documents. 
        Analyze the following legal text and identify any important clauses such as:
        1. Non-compete clauses
        2. Confidentiality clauses
        3. Termination conditions
        4. Limitations of liability
        5. Indemnification provisions
        6. Payment terms
        7. Warranty disclaimers
        8. Intellectual property rights
        9. Governing law clauses
        10. Dispute resolution mechanisms
        
        For each clause you detect, provide:
        1. The type of clause
        2. A brief explanation of its implications
        3. The level of potential risk (Low, Medium, High) if applicable
        
        Text to analyze:
        {text}
        
        Identified clauses:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"]
        )
        
        # Create and return the chain
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
        )