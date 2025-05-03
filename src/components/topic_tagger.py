import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import hdbscan
import umap
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from src.exception import CustomException
from src.logger import logging
from src.utils.common import save_object, load_object


class TopicTagger:
    """
    Component for automatic topic tagging of documents
    """
    def __init__(
        self, 
        method: str = "embeddings",
        n_clusters: int = 5,
        model_dir: str = "models"
    ):
        """
        Initialize topic tagger
        
        Args:
            method: Method for topic tagging ('tfidf' or 'embeddings')
            n_clusters: Number of clusters/topics to identify
            model_dir: Directory to save models
        """
        self.method = method
        self.n_clusters = n_clusters
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "embeddings":
            try:
                # Use sentence-transformers for better semantic understanding
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                # UMAP for dimensionality reduction before clustering
                self.reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
                # HDBSCAN for clustering
                self.clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', 
                                                 cluster_selection_method='eom', prediction_data=True)
            except Exception as e:
                logging.error(f"Error initializing embedding model: {e}")
                # Fallback to TF-IDF if embeddings fail
                logging.info("Falling back to TF-IDF method for topic tagging")
                self.method = "tfidf"
                self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tfidf' or 'embeddings'")
    
    def _extract_text(self, documents: List[Document]) -> List[str]:
        """
        Extract text content from documents
        """
        return [doc.page_content for doc in documents]
    
    def _get_representative_terms(self, 
                                 feature_names: List[str], 
                                 cluster_centers: np.ndarray, 
                                 n_terms: int = 5) -> Dict[int, List[str]]:
        """
        Get most representative terms for each cluster from TF-IDF
        """
        terms_per_cluster = {}
        for i in range(len(cluster_centers)):
            # Get indices of top n terms for this cluster
            indices = np.argsort(cluster_centers[i])[:-(n_terms+1):-1]
            terms_per_cluster[i] = [feature_names[j] for j in indices]
        
        return terms_per_cluster
    
    def _get_cluster_labels(self, terms_per_cluster: Dict[int, List[str]]) -> Dict[int, str]:
        """
        Convert representative terms to human-readable cluster labels
        """
        cluster_labels = {}
        for cluster_id, terms in terms_per_cluster.items():
            label = f"Topic {cluster_id}: {', '.join(terms[:3])}"
            cluster_labels[cluster_id] = label
        
        return cluster_labels
    
    def fit(self, documents: List[Document]) -> Dict[int, str]:
        """
        Fit the topic tagging model and return cluster labels
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary mapping cluster IDs to topic labels
        """
        try:
            texts = self._extract_text(documents)
            
            if self.method == "tfidf":
                # TF-IDF + K-Means approach
                vectors = self.vectorizer.fit_transform(texts)
                self.clusterer.fit(vectors)
                
                # Get cluster centers and feature names
                feature_names = self.vectorizer.get_feature_names_out()
                cluster_centers = self.clusterer.cluster_centers_
                
                # Get representative terms for each cluster
                terms_per_cluster = self._get_representative_terms(feature_names, cluster_centers)
                
                # Create human-readable cluster labels
                cluster_labels = self._get_cluster_labels(terms_per_cluster)
                
                # Save models
                save_object(os.path.join(self.model_dir, "vectorizer.pkl"), self.vectorizer)
                save_object(os.path.join(self.model_dir, "clusterer.pkl"), self.clusterer)
                save_object(os.path.join(self.model_dir, "cluster_labels.pkl"), cluster_labels)
                
                return cluster_labels
                
            elif self.method == "embeddings":
                # Sentence embeddings + UMAP + HDBSCAN approach
                embeddings = self.embedding_model.encode(texts)
                
                # Reduce dimensions with UMAP
                reduced_embeddings = self.reducer.fit_transform(embeddings)
                
                # Cluster with HDBSCAN
                self.clusterer.fit(reduced_embeddings)
                
                # Create cluster labels
                unique_clusters = set(self.clusterer.labels_)
                if -1 in unique_clusters:  # Remove noise cluster
                    unique_clusters.remove(-1)
                
                # For each cluster, find most central documents
                cluster_labels = {}
                for cluster_id in unique_clusters:
                    cluster_mask = self.clusterer.labels_ == cluster_id
                    cluster_texts = [texts[i] for i, mask in enumerate(cluster_mask) if mask]
                    
                    # Simple approach: use first few words from most central document
                    if cluster_texts:
                        central_doc = cluster_texts[0]
                        words = ' '.join(central_doc.split()[:10])
                        cluster_labels[cluster_id] = f"Topic {cluster_id}: {words}..."
                    else:
                        cluster_labels[cluster_id] = f"Topic {cluster_id}"
                
                # Save models
                save_object(os.path.join(self.model_dir, "embedding_model.pkl"), self.embedding_model)
                save_object(os.path.join(self.model_dir, "reducer.pkl"), self.reducer)
                save_object(os.path.join(self.model_dir, "clusterer.pkl"), self.clusterer)
                save_object(os.path.join(self.model_dir, "cluster_labels.pkl"), cluster_labels)
                
                return cluster_labels
                
        except Exception as e:
            logging.error(f"Error fitting topic model: {e}")
            raise CustomException(e, sys)
    
    def predict(self, documents: List[Document]) -> List[str]:
        """
        Predict topics for new documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of topic labels for each document
        """
        try:
            texts = self._extract_text(documents)
            cluster_labels = load_object(os.path.join(self.model_dir, "cluster_labels.pkl"))
            
            if self.method == "tfidf":
                # Transform new documents using fitted vectorizer
                vectorizer = load_object(os.path.join(self.model_dir, "vectorizer.pkl"))
                clusterer = load_object(os.path.join(self.model_dir, "clusterer.pkl"))
                
                vectors = vectorizer.transform(texts)
                cluster_ids = clusterer.predict(vectors)
                
                # Map cluster IDs to labels
                predicted_labels = [cluster_labels.get(cluster_id, f"Unknown Topic {cluster_id}") 
                                    for cluster_id in cluster_ids]
                
                return predicted_labels
                
            elif self.method == "embeddings":
                # Transform using sentence embeddings
                embedding_model = load_object(os.path.join(self.model_dir, "embedding_model.pkl"))
                reducer = load_object(os.path.join(self.model_dir, "reducer.pkl"))
                clusterer = load_object(os.path.join(self.model_dir, "clusterer.pkl"))
                
                embeddings = embedding_model.encode(texts)
                reduced_embeddings = reducer.transform(embeddings)
                
                # Predict using HDBSCAN
                cluster_ids, _ = hdbscan.approximate_predict(clusterer, reduced_embeddings)
                
                # Map cluster IDs to labels
                predicted_labels = []
                for cluster_id in cluster_ids:
                    if cluster_id == -1:  # Noise
                        predicted_labels.append("Uncategorized")
                    else:
                        predicted_labels.append(cluster_labels.get(cluster_id, f"Unknown Topic {cluster_id}"))
                
                return predicted_labels
                
        except Exception as e:
            logging.error(f"Error predicting topics: {e}")
            raise CustomException(e, sys)
    
    def tag_documents(self, documents: List[Document]) -> List[Document]:
        """
        Tag documents with topics and return updated documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Documents with topic metadata added
        """
        try:
            # First fit the model
            self.fit(documents)
            
            # Then predict topics
            topic_labels = self.predict(documents)
            
            # Add topics to document metadata
            for doc, topic in zip(documents, topic_labels):
                doc.metadata["topic"] = topic
            
            logging.info(f"Tagged {len(documents)} documents with topics")
            return documents
            
        except Exception as e:
            logging.error(f"Error tagging documents: {e}")
            raise CustomException(e, sys)