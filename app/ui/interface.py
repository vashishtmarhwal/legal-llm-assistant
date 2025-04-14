import streamlit as st
import pandas as pd
import io
import os
import json
from typing import List, Dict, Any, Optional

from app.core.document_processor import DocumentProcessor
from app.services.extraction import InformationExtractor
from app.services.summarization import DocumentSummarizer
from app.services.qa import QuestionAnswerer
from app.services.clause_detection import ClauseDetector


class LegalAssistantUI:
    """
    Web interface for the Legal Assistant application using Streamlit.
    """
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        extractor: InformationExtractor,
        summarizer: DocumentSummarizer,
        qa_system: QuestionAnswerer,
        clause_detector: ClauseDetector
    ):
        """
        Initialize the UI with required service components.
        
        Args:
            document_processor: Component for processing documents
            extractor: Component for extracting information
            summarizer: Component for document summarization
            qa_system: Component for answering questions
            clause_detector: Component for detecting legal clauses
        """
        self.document_processor = document_processor
        self.extractor = extractor
        self.summarizer = summarizer
        self.qa_system = qa_system
        self.clause_detector = clause_detector
        
        # Initialize session state variables
        if 'documents' not in st.session_state:
            st.session_state.documents = {}
        if 'current_doc_id' not in st.session_state:
            st.session_state.current_doc_id = None
        if 'detected_clauses' not in st.session_state:
            st.session_state.detected_clauses = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def run(self):
        """
        Run the Streamlit application.
        """
        st.set_page_config(
            page_title="Legal Document Assistant",
            page_icon="⚖️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("Legal Document Assistant")
        
        # Create sidebar for document upload and selection
        self._create_sidebar()
        
        # If no document is loaded, show welcome screen
        if not st.session_state.current_doc_id:
            self._show_welcome_screen()
            return
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs([
            "Document Overview", 
            "Clause Analysis", 
            "Ask Questions",
            "Compare Documents"
        ])
        
        # Tab 1: Document Overview
        with tab1:
            self._document_overview_tab()
        
        # Tab 2: Clause Analysis
        with tab2:
            self._clause_analysis_tab()
        
        # Tab 3: Ask Questions
        with tab3:
            self._qa_tab()
        
        # Tab 4: Compare Documents
        with tab4:
            self._document_comparison_tab()
    
    def _create_sidebar(self):
        """
        Create the sidebar for document management.
        """
        with st.sidebar:
            st.header("Document Management")
            
            # Document upload
            uploaded_file = st.file_uploader(
                "Upload a legal document",
                type=["pdf", "docx", "txt"],
                help="Upload a legal document in PDF, Word, or text format"
            )
            
            if uploaded_file is not None:
                # Process the uploaded document
                doc_id = self._process_uploaded_document(uploaded_file)
                if doc_id:
                    st.success(f"Document '{uploaded_file.name}' uploaded successfully!")
            
            # Document selection
            if st.session_state.documents:
                st.subheader("Your Documents")
                doc_names = {
                    doc_id: doc_info["name"] 
                    for doc_id, doc_info in st.session_state.documents.items()
                }
                
                selected_doc_name = st.selectbox(
                    "Select a document to analyze",
                    options=list(doc_names.values()),
                    index=0 if st.session_state.current_doc_id else 0
                )
                
                # Find the doc_id for the selected name
                selected_doc_id = next(
                    (doc_id for doc_id, name in doc_names.items() if name == selected_doc_name),
                    None
                )
                
                if selected_doc_id and selected_doc_id != st.session_state.current_doc_id:
                    st.session_state.current_doc_id = selected_doc_id
                    st.session_state.detected_clauses = []  # Reset clauses for new document
                    st.session_state.chat_history = []  # Reset chat history
                    st.experimental_rerun()
            
            # Show document stats
            if st.session_state.current_doc_id:
                self._show_document_stats()
    
    def _process_uploaded_document(self, uploaded_file) -> Optional[str]:
        """
        Process an uploaded document and store it in session state.
        
        Args:
            uploaded_file: The uploaded file object
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            # Read file content
            file_content = uploaded_file.read()
            
            # Process document based on file type
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Process the document
            processed_doc = self.document_processor.process_document(
                file_content, 
                file_type=file_extension[1:],  # Remove the dot
                filename=uploaded_file.name
            )
            
            # Generate a unique document ID
            import uuid
            doc_id = str(uuid.uuid4())
            
            # Store in session state
            st.session_state.documents[doc_id] = {
                "name": uploaded_file.name,
                "content": processed_doc["content"],
                "metadata": processed_doc["metadata"],
                "sections": processed_doc.get("sections", []),
                "summary": None,  # Will be populated when requested
                "upload_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Set as current document
            st.session_state.current_doc_id = doc_id
            
            return doc_id
        
        except Exception as e:
            st.error(f"Error processing document: {e}")
            return None
    
    def _show_document_stats(self):
        """
        Display statistics about the current document.
        """
        if not st.session_state.current_doc_id:
            return
        
        doc_info = st.session_state.documents[st.session_state.current_doc_id]
        
        st.sidebar.subheader("Document Stats")
        st.sidebar.markdown(f"**Name:** {doc_info['name']}")
        st.sidebar.markdown(f"**Uploaded:** {doc_info['upload_time']}")
        
        # Calculate and show document stats
        word_count = len(doc_info['content'].split())
        page_count = doc_info['metadata'].get('page_count', 'N/A')
        
        st.sidebar.markdown(f"**Words:** {word_count}")
        st.sidebar.markdown(f"**Pages:** {page_count}")
        
        # Add download button for the processed text
        text_io = io.StringIO(doc_info['content'])
        st.sidebar.download_button(
            label="Download Text",
            data=text_io,
            file_name=f"{os.path.splitext(doc_info['name'])[0]}.txt",
            mime="text/plain"
        )
    
    def _show_welcome_screen(self):
        """
        Display the welcome screen when no document is loaded.
        """
        st.markdown("""
        ## Welcome to the Legal Document Assistant

        This tool helps you analyze legal documents with AI-powered features:

        - **Document Overview**: Get summaries and key information extraction
        - **Clause Analysis**: Identify and analyze legal clauses
        - **Question Answering**: Ask questions about your document
        - **Document Comparison**: Compare multiple legal documents

        To get started, upload a document using the sidebar on the left.
        """)
        
        # Sample documents section (optional)
        st.subheader("Try with a sample document")
        if st.button("Load Sample Contract"):
            # Load a sample contract document
            self._load_sample_document("sample_contract.pdf")
    
    def _load_sample_document(self, sample_filename):
        """
        Load a sample document for demonstration purposes.
        
        Args:
            sample_filename: Filename of the sample document to load
        """
        try:
            # Path to sample documents
            sample_path = os.path.join("data", "samples", sample_filename)
            
            with open(sample_path, "rb") as f:
                file_content = f.read()
            
            # Process the sample document
            processed_doc = self.document_processor.process_document(
                file_content, 
                file_type=os.path.splitext(sample_filename)[1][1:],
                filename=sample_filename
            )
            
            # Generate a unique document ID
            import uuid
            doc_id = str(uuid.uuid4())
            
            # Store in session state
            st.session_state.documents[doc_id] = {
                "name": f"Sample: {sample_filename}",
                "content": processed_doc["content"],
                "metadata": processed_doc["metadata"],
                "sections": processed_doc.get("sections", []),
                "summary": None,
                "upload_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Set as current document
            st.session_state.current_doc_id = doc_id
            
            st.experimental_rerun()
        
        except Exception as e:
            st.error(f"Error loading sample document: {e}")
    
    def _document_overview_tab(self):
        """
        Create the document overview tab with summary and key information.
        """
        st.header("Document Overview")
        
        doc_id = st.session_state.current_doc_id
        if not doc_id:
            st.warning("Please upload or select a document first.")
            return
        
        doc_info = st.session_state.documents[doc_id]
        
        # Generate summary if not already available
        if not doc_info.get("summary"):
            with st.spinner("Generating document summary..."):
                summary = self.summarizer.summarize(doc_info["content"])
                st.session_state.documents[doc_id]["summary"] = summary
        
        # Display summary
        st.subheader("Document Summary")
        st.write(doc_info["summary"])
        
        # Extract key information
        st.subheader("Key Information")
        
        with st.spinner("Extracting key information..."):
            key_info = self.extractor.extract_key_information(doc_info["content"])
            
            # Display in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Parties")
                if "parties" in key_info:
                    for party in key_info["parties"]:
                        st.write(f"- {party}")
                else:
                    st.write("No parties identified.")
                
                st.markdown("### Dates")
                if "dates" in key_info:
                    for date_type, date_value in key_info["dates"].items():
                        st.write(f"- **{date_type.title()}**: {date_value}")
                else:
                    st.write("No significant dates identified.")
            
            with col2:
                st.markdown("### Key Terms")
                if "key_terms" in key_info:
                    for term_type, term_value in key_info["key_terms"].items():
                        st.write(f"- **{term_type.title()}**: {term_value}")
                else:
                    st.write("No key terms identified.")
        
        # Document full text with expandable section
        with st.expander("View Full Document Text"):
            st.text_area(
                "Document Content",
                value=doc_info["content"],
                height=400,
                disabled=True
            )
    
    def _clause_analysis_tab(self):
        """
        Create the clause analysis tab for detecting and analyzing legal clauses.
        """
        st.header("Clause Analysis")
        
        doc_id = st.session_state.current_doc_id
        if not doc_id:
            st.warning("Please upload or select a document first.")
            return
        
        doc_info = st.session_state.documents[doc_id]
        
        # Detect clauses if not already done
        if not st.session_state.detected_clauses:
            if st.button("Detect Legal Clauses"):
                with st.spinner("Detecting legal clauses..."):
                    detected_clauses = self.clause_detector.detect_clauses(doc_info["content"])
                    st.session_state.detected_clauses = detected_clauses
                    st.experimental_rerun()
        else:
            # Show detected clauses
            st.write(f"Found {len(st.session_state.detected_clauses)} legal clauses")
            
            # Group clauses by type
            clause_types = {}
            for clause in st.session_state.detected_clauses:
                clause_type = clause["type"]
                if clause_type not in clause_types:
                    clause_types[clause_type] = []
                clause_types[clause_type].append(clause)
            
            # Create tabs for each clause type
            if clause_types:
                clause_tabs = st.tabs(list(clause_types.keys()))
                
                for i, (clause_type, clauses) in enumerate(clause_types.items()):
                    with clause_tabs[i]:
                        for j, clause in enumerate(clauses):
                            with st.expander(f"{clause_type.title()} Clause {j+1} - Risk: {clause['risk_assessment'].upper()}"):
                                st.markdown("**Clause Text:**")
                                st.markdown(f"```\n{clause['text']}\n```")
                                
                                st.markdown(f"**Risk Assessment:** {clause['risk_assessment'].upper()}")
                                st.markdown(f"**Comments:** {clause['comments']}")
                                
                                # Add analyze button
                                if st.button(f"Analyze {clause_type} Clause {j+1}", key=f"analyze_{clause_type}_{j}"):
                                    with st.spinner("Analyzing clause..."):
                                        suggestions = self.clause_detector.suggest_improvements(
                                            clause['text'], clause['type']
                                        )
                                        
                                        st.markdown("### Suggested Improvements")
                                        for suggestion in suggestions:
                                            st.markdown(f"- {suggestion}")
                                        
                                        # Extract key terms
                                        terms = self.clause_detector.extract_key_terms(clause['text'])
                                        
                                        st.markdown("### Key Terms")
                                        for term_key, term_value in terms.items():
                                            if term_key != "clause_type" and term_key != "error":
                                                st.markdown(f"- **{term_key.replace('_', ' ').title()}**: {term_value}")
            else:
                st.info("No legal clauses were detected in this document.")
        
        # Reset button
        if st.session_state.detected_clauses:
            if st.button("Reset Clause Detection"):
                st.session_state.detected_clauses = []
                st.experimental_rerun()
    
    def _qa_tab(self):
        """
        Create the question answering tab.
        """
        st.header("Ask Questions")
        
        doc_id = st.session_state.current_doc_id
        if not doc_id:
            st.warning("Please upload or select a document first.")
            return
        
        doc_info = st.session_state.documents[doc_id]
        
        # Input for questions
        question = st.text_input(
            "Ask a question about the document:",
            placeholder="E.g., What are the termination conditions?"
        )
        
        # Show suggested questions
        st.caption("Suggested questions:")
        suggestion_cols = st.columns(3)
        
        suggested_questions = [
            "What are the key obligations?",
            "How can this contract be terminated?",
            "What's the governing law?"
        ]
        
        for i, suggested_q in enumerate(suggested_questions):
            with suggestion_cols[i % 3]:
                if st.button(suggested_q, key=f"suggested_q_{i}"):
                    question = suggested_q
                    # Need to re-run to update the text input
                    st.experimental_rerun()
        
        # Display chat history
        st.subheader("Conversation")
        
        chat_container = st.container()
        
        # Process question
        if question and question.strip():
            # Add question to history
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Get answer from QA system
            with st.spinner("Thinking..."):
                answer = self.qa_system.answer_question(question, doc_info["content"])
                
                # Add answer to history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer["answer"],
                    "sources": answer.get("sources", [])
                })
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Assistant:** {message['content']}")
                    
                    # Show sources if available
                    if "sources" in message and message["sources"]:
                        with st.expander("Sources"):
                            for source in message["sources"]:
                                st.markdown(f"```\n{source}\n```")
                
                st.markdown("---")
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("Clear Conversation"):
                st.session_state.chat_history = []
                st.experimental_rerun()
    
    def _document_comparison_tab(self):
        """
        Create the document comparison tab.
        """
        st.header("Document Comparison")
        
        # Check if we have at least 2 documents
        if len(st.session_state.documents) < 2:
            st.warning("You need to upload at least 2 documents to use the comparison feature.")
            return
        
        # Document selection
        st.subheader("Select Documents to Compare")
        
        doc_names = {
            doc_id: doc_info["name"] 
            for doc_id, doc_info in st.session_state.documents.items()
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            doc1_name = st.selectbox(
                "First Document",
                options=list(doc_names.values()),
                index=0
            )
            
            # Find the doc_id for the selected name
            doc1_id = next(
                (doc_id for doc_id, name in doc_names.items() if name == doc1_name),
                None
            )
        
        with col2:
            # Filter out the first document from options
            remaining_docs = {
                doc_id: name for doc_id, name in doc_names.items() if name != doc1_name
            }
            
            doc2_name = st.selectbox(
                "Second Document",
                options=list(remaining_docs.values()),
                index=0
            )
            
            # Find the doc_id for the selected name
            doc2_id = next(
                (doc_id for doc_id, name in doc_names.items() if name == doc2_name),
                None
            )
        
        # Compare button
        if st.button("Compare Documents"):
            if doc1_id and doc2_id:
                with st.spinner("Comparing documents..."):
                    self._show_document_comparison(doc1_id, doc2_id)
    
    def _show_document_comparison(self, doc1_id: str, doc2_id: str):
        """
        Display comparison between two documents.
        
        Args:
            doc1_id: ID of the first document
            doc2_id: ID of the second document
        """
        doc1_info = st.session_state.documents[doc1_id]
        doc2_info = st.session_state.documents[doc2_id]
        
        # Get document content
        doc1_content = doc1_info["content"]
        doc2_content = doc2_info["content"]
        
        # Compare using the clause detector
        st.subheader("Clause Comparison")
        
        # Detect clauses in both documents
        with st.spinner("Detecting clauses in both documents..."):
            doc1_clauses = self.clause_detector.detect_clauses(doc1_content)
            doc2_clauses = self.clause_detector.detect_clauses(doc2_content)
            
            # Group clauses by type
            doc1_by_type = self._group_clauses_by_type(doc1_clauses)
            doc2_by_type = self._group_clauses_by_type(doc2_clauses)
            
            # Find common clause types
            common_types = set(doc1_by_type.keys()) & set(doc2_by_type.keys())
            
            if not common_types:
                st.info("No common clause types found for direct comparison.")
            else:
                # Create comparison for each common type
                for clause_type in sorted(common_types):
                    st.markdown(f"### {clause_type.title()} Clauses")
                    
                    # Take the first clause of each type for simplicity
                    doc1_clause = doc1_by_type[clause_type][0]["text"]
                    doc2_clause = doc2_by_type[clause_type][0]["text"]
                    
                    # Compare the clauses
                    comparison = self.clause_detector.compare_clauses(doc1_clause, doc2_clause)
                    
                    # Display side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{doc1_info['name']}**")
                        st.markdown(f"```\n{doc1_clause[:500]}{'...' if len(doc1_clause) > 500 else ''}\n```")
                    
                    with col2:
                        st.markdown(f"**{doc2_info['name']}**")
                        st.markdown(f"```\n{doc2_clause[:500]}{'...' if len(doc2_clause) > 500 else ''}\n```")
                    
                    # Show comparison results
                    st.markdown("#### Comparison Results")
                    
                    st.markdown("**Key Differences:**")
                    for diff in comparison.get("key_differences", []):
                        st.markdown(f"- {diff}")
                    
                    preferred = comparison.get("preferred_clause", "neither")
                    if preferred == "1":
                        preferred_doc = doc1_info['name']
                    elif preferred == "2":
                        preferred_doc = doc2_info['name']
                    else:
                        preferred_doc = "Neither document"
                    
                    st.markdown(f"**Preferred Clause:** {preferred_doc}")
                    
                    st.markdown("**Concerns:**")
                    for concern in comparison.get("concerns", []):
                        st.markdown(f"- {concern}")
                    
                    st.markdown("**Recommendations:**")
                    for rec in comparison.get("recommendations", []):
                        st.markdown(f"- {rec}")
                    
                    st.markdown("---")
        
        # Overall document comparison
        st.subheader("Overall Document Comparison")
        
        with st.spinner("Generating overall comparison..."):
            # Use the summarizer to compare documents
            comparison_prompt = f"""
            Compare the following two legal documents and provide a summary of key differences.
            First document: {doc1_content[:5000]}
            Second document: {doc2_content[:5000]}
            """
            
            comparison_summary = self.summarizer.summarize(comparison_prompt)
            st.markdown(comparison_summary)
    
    def _group_clauses_by_type(self, clauses: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group clauses by their type.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            Dictionary with clause types as keys and lists of clauses as values
        """
        result = {}
        for clause in clauses:
            clause_type = clause["type"]
            if clause_type not in result:
                result[clause_type] = []
            result[clause_type].append(clause)
        return result


def create_ui(
    document_processor: DocumentProcessor,
    extractor: InformationExtractor,
    summarizer: DocumentSummarizer,
    qa_system: QuestionAnswerer,
    clause_detector: ClauseDetector
):
    """
    Factory function to create and run the UI.
    
    Args:
        document_processor: Component for processing documents
        extractor: Component for extracting information
        summarizer: Component for document summarization
        qa_system: Component for answering questions
        clause_detector: Component for detecting legal clauses
        
    Returns:
        The UI instance
    """
    ui = LegalAssistantUI(
        document_processor=document_processor,
        extractor=extractor,
        summarizer=summarizer,
        qa_system=qa_system,
        clause_detector=clause_detector
    )
    return ui
