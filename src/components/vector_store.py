import os
import sys
from typing import Dict, List, Optional, Union

from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils.common import ensure_directory_exists


from langchain_core.embeddings import Embeddings

class EmbeddingModel(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            logging.info(f"Initialized embedding model: {model_name}")
        except Exception as e:
            logging.error(f"Error initializing embedding model: {e}")
            raise CustomException(e, sys)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logging.error(f"Error embedding documents: {e}")
            raise CustomException(e, sys)

    def embed_query(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logging.error(f"Error embedding query: {e}")
            raise CustomException(e, sys)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Required by Langchain to treat this object as a callable embedding function
        """
        return self.embed_documents(texts)


class VectorStore:
    """
    Component for creating and managing a vector store
    """
    def __init__(
        self,
        persist_directory: str = "vectorstore",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        store_type: str = "faiss"
    ):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist vector store
            embedding_model_name: Name of the embedding model to use
            store_type: Type of vector store ('faiss' or 'chroma')
        """
        self.persist_directory = persist_directory
        ensure_directory_exists(persist_directory)
        self.store_type = store_type
        
        # Initialize embedding model
        self.embeddings = EmbeddingModel(embedding_model_name)
        
        self.vector_store = None
        logging.info(f"Initialized {store_type} vector store with {embedding_model_name} embeddings")
    
    def create_or_load(self, documents: Optional[List[Document]] = None) -> Union[FAISS, Chroma]:
        """
        Create a new vector store or load an existing one
        
        Args:
            documents: List of Document objects to add to vector store (if creating new)
            
        Returns:
            Vector store instance
        """
        try:
            if self.store_type == "faiss":
                if documents and len(documents) > 0:
                    # Create new vector store
                    self.vector_store = FAISS.from_documents(documents, self.embeddings)
                    # Save to disk
                    self.vector_store.save_local(self.persist_directory)
                    logging.info(f"Created FAISS vector store with {len(documents)} documents")
                elif os.path.exists(os.path.join(self.persist_directory, "index.faiss")):
                    # Load existing vector store
                    self.vector_store = FAISS.load_local(self.persist_directory, self.embeddings)
                    logging.info(f"Loaded existing FAISS vector store from {self.persist_directory}")
                else:
                    raise FileNotFoundError(f"No existing vector store found and no documents provided")
            
            elif self.store_type == "chroma":
                if documents and len(documents) > 0:
                    # Create new vector store
                    self.vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                    # Save to disk
                    self.vector_store.persist()
                    logging.info(f"Created Chroma vector store with {len(documents)} documents")
                elif os.path.exists(self.persist_directory):
                    # Load existing vector store
                    self.vector_store = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings
                    )
                    logging.info(f"Loaded existing Chroma vector store from {self.persist_directory}")
                else:
                    raise FileNotFoundError(f"No existing vector store found and no documents provided")
            
            else:
                raise ValueError(f"Unknown store type: {self.store_type}")
            
            return self.vector_store
            
        except Exception as e:
            logging.error(f"Error creating/loading vector store: {e}")
            raise CustomException(e, sys)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to an existing vector store
        
        Args:
            documents: List of Document objects to add
        """
        try:
            if not self.vector_store:
                # Create vector store if it doesn't exist
                self.create_or_load(documents)
                return
            
            if self.store_type == "faiss":
                self.vector_store.add_documents(documents)
                # Save to disk
                self.vector_store.save_local(self.persist_directory)
                logging.info(f"Added {len(documents)} documents to FAISS vector store")
            
            elif self.store_type == "chroma":
                self.vector_store.add_documents(documents)
                # Save to disk
                self.vector_store.persist()
                logging.info(f"Added {len(documents)} documents to Chroma vector store")
            
        except Exception as e:
            logging.error(f"Error adding documents to vector store: {e}")
            raise CustomException(e, sys)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Query text
            k: Number of results to return
            filter_dict: Dictionary of metadata filters
            
        Returns:
            List of Document objects
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Call create_or_load first.")
            
            # Handle filters differently based on vector store type
            if filter_dict:
                if self.store_type == "faiss":
                    # FAISS requires manual filtering after search
                    raw_results = self.vector_store.similarity_search(query, k=k*3)  # Get more results to filter
                    
                    # Apply filters manually
                    filtered_results = []
                    for doc in raw_results:
                        match = True
                        for key, value in filter_dict.items():
                            if key not in doc.metadata or doc.metadata[key] != value:
                                match = False
                                break
                        if match:
                            filtered_results.append(doc)
                    
                    results = filtered_results[:k]  # Limit to k results
                    
                elif self.store_type == "chroma":
                    # Chroma supports filter dictionaries natively
                    results = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            else:
                # No filters, just do regular search
                results = self.vector_store.similarity_search(query, k=k)
            
            logging.info(f"Found {len(results)} documents matching query")
            return results
            
        except Exception as e:
            logging.error(f"Error performing similarity search: {e}")
            raise CustomException(e, sys)