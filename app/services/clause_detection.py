from typing import List, Dict, Any, Tuple
import re
import json
import numpy as np
from collections import defaultdict

from app.core.llm_manager import LLMManager
from app.core.vector_store import VectorStore


class ClauseDetector:
    """
    Responsible for detecting, classifying, and analyzing legal clauses in documents.
    """
    
    def __init__(self, llm_manager: LLMManager, vector_store: VectorStore):
        """
        Initialize the ClauseDetector with required dependencies.
        
        Args:
            llm_manager: An instance of LLMManager for text processing
            vector_store: An instance of VectorStore for similarity searches
        """
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.common_clause_types = [
            "indemnification", "limitation of liability", "confidentiality", 
            "termination", "governing law", "force majeure", "warranties",
            "payment terms", "intellectual property", "non-compete", 
            "arbitration", "amendment", "assignment", "severability"
        ]
        
        # Load clause patterns and examples if vector store is populated
        self._initialize_clause_patterns()
    
    def _initialize_clause_patterns(self):
        """
        Initialize common clause patterns and examples from the vector store if available.
        """
        try:
            self.clause_examples = self.vector_store.get_collection("clause_examples")
            if not self.clause_examples:
                # Create default examples if none exist
                self._create_default_clause_examples()
        except Exception as e:
            print(f"Warning: Could not initialize clause patterns: {e}")
            self.clause_examples = {}
    
    def _create_default_clause_examples(self):
        """
        Create and store default clause examples if none exist in the vector store.
        """
        default_examples = {
            "indemnification": [
                "Each party shall indemnify, defend and hold harmless the other party from and against any claims...",
                "The Client agrees to indemnify and hold harmless the Company against all losses, damages, costs..."
            ],
            "limitation of liability": [
                "In no event shall either party be liable for any indirect, incidental, special or consequential damages...",
                "The Company's total liability under this Agreement shall not exceed the amounts paid by Client..."
            ],
            # More default examples for other clause types would be added here
        }
        
        # Store these in vector store for future use
        embeddings = {}
        for clause_type, examples in default_examples.items():
            for example in examples:
                embedding = self.llm_manager.get_embedding(example)
                if clause_type not in embeddings:
                    embeddings[clause_type] = []
                embeddings[clause_type].append((example, embedding))
        
        # Store in vector store
        self.vector_store.create_collection("clause_examples", embeddings)
        self.clause_examples = embeddings
    
    def detect_clauses(self, document_text: str) -> List[Dict[str, Any]]:
        """
        Detect legal clauses within a document.
        
        Args:
            document_text: The full text of the document to analyze
            
        Returns:
            A list of dictionaries containing clause information:
            [
                {
                    "type": "indemnification",
                    "text": "The full text of the clause...",
                    "start_index": 1250,
                    "end_index": 1500,
                    "risk_assessment": "medium",
                    "comments": "Standard indemnification clause with mutual protection"
                },
                ...
            ]
        """
        # First, segment the document into potential clause paragraphs
        paragraphs = self._segment_document(document_text)
        
        detected_clauses = []
        
        for idx, para in enumerate(paragraphs):
            # Skip very short paragraphs as they're unlikely to be full clauses
            if len(para.strip()) < 50:
                continue
            
            # Detect if paragraph contains a legal clause
            clause_info = self._classify_clause(para)
            
            if clause_info["type"]:
                # Calculate indices in the original document
                start_idx = document_text.find(para)
                end_idx = start_idx + len(para) if start_idx != -1 else -1
                
                # Get risk assessment and comments
                risk_info = self._assess_clause_risk(para, clause_info["type"])
                
                detected_clauses.append({
                    "type": clause_info["type"],
                    "text": para,
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "confidence": clause_info["confidence"],
                    "risk_assessment": risk_info["risk_level"],
                    "comments": risk_info["comments"]
                })
        
        return detected_clauses
    
    def _segment_document(self, document_text: str) -> List[str]:
        """
        Segment document into potential clause paragraphs.
        
        Args:
            document_text: Complete document text
            
        Returns:
            List of text paragraphs
        """
        # Simple approach: split by double newlines, considering them paragraph breaks
        paragraphs = re.split(r'\n\s*\n', document_text)
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _classify_clause(self, text: str) -> Dict[str, Any]:
        """
        Classify a text segment as a specific type of legal clause.
        
        Args:
            text: Text segment to classify
            
        Returns:
            Dictionary with clause type and confidence score
        """
        # First, quick rule-based check for common clause headers
        for clause_type in self.common_clause_types:
            pattern = rf'\b{clause_type}\b'
            if re.search(pattern, text.lower(), re.IGNORECASE):
                return {"type": clause_type, "confidence": 0.9}
        
        # If no obvious match, use semantic similarity with the vector store
        embedding = self.llm_manager.get_embedding(text)
        
        # Get the most similar clause examples from the vector store
        best_match = None
        highest_similarity = -1
        
        # Check similarity against our examples
        for clause_type, examples in self.clause_examples.items():
            for example_text, example_embedding in examples:
                similarity = self._calculate_similarity(embedding, example_embedding)
                if similarity > highest_similarity and similarity > 0.7:  # Threshold
                    highest_similarity = similarity
                    best_match = clause_type
        
        # If no good match, use LLM as fallback
        if not best_match:
            prompt = f"""
            Analyze the following text and determine if it represents a specific type of legal clause.
            If yes, identify the type of clause (e.g., indemnification, limitation of liability, etc.).
            If it's not a legal clause or you're unsure, respond with "unknown".
            
            Text: "{text[:500]}..."
            
            Clause type:
            """
            
            result = self.llm_manager.generate_text(prompt)
            
            # Clean up and normalize the result
            result = result.strip().lower()
            if result != "unknown" and any(clause_type in result for clause_type in self.common_clause_types):
                for clause_type in self.common_clause_types:
                    if clause_type in result:
                        best_match = clause_type
                        highest_similarity = 0.7  # Default confidence for LLM classification
                        break
        
        return {
            "type": best_match if best_match else "other",
            "confidence": highest_similarity if best_match else 0.0
        }
    
    def _calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings."""
        # Convert to numpy arrays if they aren't already
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)
        
        # Calculate cosine similarity
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def _assess_clause_risk(self, clause_text: str, clause_type: str) -> Dict[str, str]:
        """
        Assess the risk level of a legal clause and provide comments.
        
        Args:
            clause_text: Text of the clause
            clause_type: Type of the clause
            
        Returns:
            Dictionary with risk level and comments
        """
        prompt = f"""
        Analyze the following {clause_type} clause and assess its risk level (low, medium, high).
        Provide a brief explanation of your assessment focusing on potential risks or benefits.
        
        Clause: "{clause_text[:1000]}..."
        
        Respond in JSON format with fields: "risk_level" and "comments".
        """
        
        try:
            response = self.llm_manager.generate_text(prompt)
            result = json.loads(response)
            return {
                "risk_level": result.get("risk_level", "medium"),
                "comments": result.get("comments", "No specific concerns identified.")
            }
        except Exception as e:
            print(f"Error in risk assessment: {e}")
            return {
                "risk_level": "medium",
                "comments": "Unable to perform detailed analysis."
            }
    
    def compare_clauses(self, clause1: str, clause2: str) -> Dict[str, Any]:
        """
        Compare two legal clauses and identify key differences.
        
        Args:
            clause1: Text of the first clause
            clause2: Text of the second clause
            
        Returns:
            Dictionary with comparison results including differences and recommendations
        """
        prompt = f"""
        Compare the following two legal clauses and identify key differences, 
        which one offers better protection, and any concerning elements.
        
        Clause 1: "{clause1[:500]}..."
        
        Clause 2: "{clause2[:500]}..."
        
        Respond in JSON format with fields: 
        1. "key_differences" (array of differences)
        2. "preferred_clause" (1, 2, or "neither")
        3. "concerns" (array of concerning elements)
        4. "recommendations" (suggestions for improvement)
        """
        
        try:
            response = self.llm_manager.generate_text(prompt)
            return json.loads(response)
        except Exception as e:
            print(f"Error in clause comparison: {e}")
            return {
                "key_differences": ["Unable to perform detailed comparison"],
                "preferred_clause": "neither",
                "concerns": ["Analysis failed"],
                "recommendations": ["Try again with shorter clauses"]
            }
    
    def suggest_improvements(self, clause_text: str, clause_type: str) -> List[str]:
        """
        Suggest improvements for a legal clause.
        
        Args:
            clause_text: Text of the clause
            clause_type: Type of the clause
            
        Returns:
            List of suggested improvements
        """
        prompt = f"""
        Analyze the following {clause_type} clause and suggest specific improvements 
        to better protect the client's interests while keeping it fair and enforceable.
        
        Clause: "{clause_text[:1000]}..."
        
        Provide 3-5 specific, actionable suggestions.
        """
        
        try:
            response = self.llm_manager.generate_text(prompt)
            
            # Extract suggestions from the response
            suggestions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or 
                            re.match(r'^\d+\.', line)):
                    suggestions.append(re.sub(r'^[-•\d\.]+\s*', '', line))
                elif suggestions and line:  # Continuation of previous suggestion
                    suggestions[-1] += ' ' + line
            
            # If no suggestions were found with the pattern matching, use the whole text
            if not suggestions and response.strip():
                suggestions = [response.strip()]
                
            return suggestions
        except Exception as e:
            print(f"Error in suggestion generation: {e}")
            return ["Unable to generate suggestions. Please try again with a shorter clause."]
    
    def extract_key_terms(self, clause_text: str) -> Dict[str, Any]:
        """
        Extract key terms and parameters from a legal clause.
        
        Args:
            clause_text: Text of the clause
            
        Returns:
            Dictionary with extracted parameters depending on clause type
        """
        # First determine the clause type if not already known
        clause_info = self._classify_clause(clause_text)
        clause_type = clause_info["type"]
        
        # Define extraction parameters based on clause type
        extraction_params = {
            "indemnification": ["indemnifying_party", "indemnified_party", "covered_scenarios", "exclusions"],
            "limitation of liability": ["liability_cap", "excluded_damages", "carve_outs"],
            "confidentiality": ["definition", "exclusions", "term", "return_requirements"],
            "payment_terms": ["payment_due", "late_payment_penalties", "currency", "payment_method"],
            # Add more clause types and their parameters as needed
        }
        
        # Get the parameters to extract for this clause type
        params_to_extract = extraction_params.get(clause_type, ["key_terms"])
        
        prompt = f"""
        Extract the following parameters from this {clause_type} clause:
        {', '.join(params_to_extract)}
        
        Clause: "{clause_text[:1000]}..."
        
        Respond in JSON format with the parameters as keys.
        """
        
        try:
            response = self.llm_manager.generate_text(prompt)
            extracted_params = json.loads(response)
            
            # Add the clause type to the results
            extracted_params["clause_type"] = clause_type
            
            return extracted_params
        except Exception as e:
            print(f"Error in key term extraction: {e}")
            return {"clause_type": clause_type, "error": "Failed to extract parameters"}