import os
import sys
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document

from src.components.document_processor import DocumentProcessor
from src.components.topic_tagger import TopicTagger
from src.components.vector_store import VectorStore
from src.components.model_interface import ModelInterface
from src.exception import CustomException
from src.logger import logging
from src.utils.common import ensure_directory_exists


class RAGPipeline:
    """
    Main pipeline for the RAG system that orchestrates all components
    """
    def __init__(
        self,
        base_dir: str = "rag_data",
        model_provider: str = "ollama",
        model_name: str = "llama2",
        vector_store_type: str = "faiss",
        api_key: Optional[str] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            base_dir: Base directory for storing data
            model_provider: Provider of the language model ('openai' or 'ollama')
            model_name: Name of the model to use
            vector_store_type: Type of vector store ('faiss' or 'chroma')
            api_key: API key for model provider (if needed)
        """
        # Create base directory
        self.base_dir = base_dir
        ensure_directory_exists(base_dir)
        
        # Document directories
        self.temp_dir = os.path.join(base_dir, "temp_docs")
        self.vector_store_dir = os.path.join(base_dir, "vector_store")
        self.model_dir = os.path.join(base_dir, "models")
        
        ensure_directory_exists(self.temp_dir)
        ensure_directory_exists(self.vector_store_dir)
        ensure_directory_exists(self.model_dir)
        
        # Initialize components
        self.document_processor = DocumentProcessor(temp_dir=self.temp_dir)
        self.topic_tagger = TopicTagger(model_dir=self.model_dir)
        self.vector_store = VectorStore(
            persist_directory=self.vector_store_dir,
            store_type=vector_store_type
        )
        self.model_interface = ModelInterface(
            model_provider=model_provider,
            model_name=model_name,
            api_key=api_key
        )
        
        # Track processed documents
        self.processed_file_paths = []
        self.document_topics = {}
        
        logging.info("RAG pipeline initialized")
    
    def process_documents(self, file_paths: List[str]) -> None:
        """
        Process documents and add them to the vector store
        
        Args:
            file_paths: List of paths to documents
        """
        try:
            # Process documents
            docs = self.document_processor.process_multiple_documents(file_paths)
            logging.info(f"Processed {len(docs)} chunks from {len(file_paths)} documents")
            
            # Tag with topics
            tagged_docs = self.topic_tagger.tag_documents(docs)
            
            # Store topics for each document
            for doc in tagged_docs:
                source = doc.metadata.get("source")
                topic = doc.metadata.get("topic")
                if source:
                    if source not in self.document_topics:
                        self.document_topics[source] = set()
                    self.document_topics[source].add(topic)
            
            # Add to vector store
            self.vector_store.create_or_load(tagged_docs)
            
            # Add to processed documents
            self.processed_file_paths.extend(file_paths)
            
            logging.info(f"Documents processed and added to vector store")
            
        except Exception as e:
            logging.error(f"Error processing documents: {e}")
            raise CustomException(e, sys)
    
    def process_uploaded_file(self, uploaded_file) -> None:
        """
        Process an uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
        """
        try:
            # Save uploaded file
            file_path = self.document_processor.save_upload(uploaded_file)
            
            # Process the file
            self.process_documents([file_path])
            
            logging.info(f"Uploaded file processed: {file_path}")
            
        except Exception as e:
            logging.error(f"Error processing uploaded file: {e}")
            raise CustomException(e, sys)
    
    def get_document_topics(self) -> Dict[str, List[str]]:
        """
        Get topics assigned to each document
        
        Returns:
            Dictionary mapping document names to lists of topics
        """
        result = {}
        for doc_name, topics in self.document_topics.items():
            result[doc_name] = list(topics)
        return result
    
    def answer_question(
        self,
        question: str,
        chat_mode: bool = True,
        filter_topic: Optional[str] = None,
        filter_source: Optional[str] = None,
        k: int = 4
    ) -> str:
        """
        Answer a question using the RAG system
        
        Args:
            question: User question
            chat_mode: Whether to use chat history
            filter_topic: Filter documents by topic
            filter_source: Filter documents by source file
            k: Number of documents to retrieve
            
        Returns:
            Answer to the question
        """
        try:
            # Build filter dictionary
            filter_dict = {}
            if filter_topic:
                filter_dict["topic"] = filter_topic
            if filter_source:
                filter_dict["source"] = filter_source
            
            # Use empty filter if no filters specified
            filter_to_use = filter_dict if filter_dict else None
            
            # Get relevant documents
            relevant_docs = self.vector_store.similarity_search(
                query=question,
                k=k,
                filter_dict=filter_to_use
            )
            
            if not relevant_docs:
                return "I couldn't find any relevant information to answer your question. Please try a different question or add more documents."
            
            # Answer question using the model
            answer = self.model_interface.answer_question(
                question=question,
                docs=relevant_docs,
                chat_mode=chat_mode
            )
            
            return answer
            
        except Exception as e:
            logging.error(f"Error answering question: {e}")
            raise CustomException(e, sys)
    
    def clear_chat_history(self) -> None:
        """
        Clear chat history
        """
        self.model_interface.clear_chat_history()
        logging.info("Chat history cleared")