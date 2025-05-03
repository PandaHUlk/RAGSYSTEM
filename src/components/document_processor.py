import os
import sys
from typing import Dict, List, Tuple, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.exception import CustomException
from src.logger import logging
from src.utils.common import ensure_directory_exists


class DocumentProcessor:
    """
    Component to handle document loading, parsing, and chunking
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temp_dir: str = "temp_docs"
    ):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
            temp_dir: Directory to store temporary files
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temp_dir = temp_dir
        ensure_directory_exists(temp_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def save_upload(self, uploaded_file) -> str:
        """
        Save an uploaded file to disk
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Path to the saved file
        """
        try:
            file_path = os.path.join(self.temp_dir, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            logging.info(f"File saved to {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Error saving uploaded file: {e}")
            raise CustomException(e, sys)
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add source filename metadata
            for doc in documents:
                doc.metadata["source"] = os.path.basename(file_path)
                doc.metadata["file_path"] = file_path
            
            logging.info(f"Loaded {len(documents)} pages from {file_path}")
            return documents
            
        except Exception as e:
            logging.error(f"Error loading PDF: {e}")
            raise CustomException(e, sys)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            logging.info(f"Split documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logging.error(f"Error splitting documents: {e}")
            raise CustomException(e, sys)
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Process a single document (load and split)
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of processed Document chunks
        """
        try:
            # Load the document
            documents = self.load_pdf(file_path)
            
            # Split into chunks
            chunks = self.split_documents(documents)
            
            return chunks
            
        except Exception as e:
            logging.error(f"Error processing document: {e}")
            raise CustomException(e, sys)
    
    def process_multiple_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of paths to documents
            
        Returns:
            List of processed Document chunks from all documents
        """
        try:
            all_chunks = []
            
            for file_path in file_paths:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
            
            logging.info(f"Processed {len(file_paths)} documents into {len(all_chunks)} chunks total")
            return all_chunks
            
        except Exception as e:
            logging.error(f"Error processing multiple documents: {e}")
            raise CustomException(e, sys)