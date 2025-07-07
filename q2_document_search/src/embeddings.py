"""
Embeddings Module
Handles text embedding generation using sentence transformers
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages text embeddings for documents and queries"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = "./cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.embeddings_cache = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def get_embeddings(self, texts: List[str], cache_key: Optional[str] = None) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not texts:
            return np.array([])
        
        # Check cache first
        if cache_key and cache_key in self.embeddings_cache:
            logger.info(f"Using cached embeddings for {cache_key}")
            return self.embeddings_cache[cache_key]
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
            
            # Cache the embeddings
            if cache_key:
                self.embeddings_cache[cache_key] = embeddings
                self._save_embeddings_to_cache(cache_key, embeddings)
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _save_embeddings_to_cache(self, cache_key: str, embeddings: np.ndarray):
        """Save embeddings to disk cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings to cache: {cache_file}")
        except Exception as e:
            logger.error(f"Error saving embeddings to cache: {e}")
    
    def _load_embeddings_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embeddings from disk cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"Loaded embeddings from cache: {cache_file}")
                return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings from cache: {e}")
        return None
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        return self.get_embeddings([query])
    
    def get_document_embeddings(self, documents: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Generate embeddings for all document chunks"""
        all_chunks = []
        chunk_mapping = {}  # Maps chunk index to document info
        
        # Collect all chunks from all documents
        for doc_idx, doc in enumerate(documents):
            if not doc.get("processed", False):
                continue
            
            chunks = doc.get("chunks", [])
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_mapping[len(all_chunks) - 1] = {
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "doc_name": doc.get("file_name", ""),
                    "chunk_text": chunk
                }
        
        if not all_chunks:
            return {}
        
        # Generate embeddings for all chunks
        embeddings = self.get_embeddings(all_chunks, cache_key="document_chunks")
        
        # Organize embeddings by document
        doc_embeddings = {}
        for doc_idx, doc in enumerate(documents):
            if not doc.get("processed", False):
                continue
            
            doc_chunk_indices = [
                idx for idx, mapping in chunk_mapping.items() 
                if mapping["doc_idx"] == doc_idx
            ]
            
            if doc_chunk_indices:
                doc_embeddings[doc.get("file_name", f"doc_{doc_idx}")] = {
                    "embeddings": embeddings[doc_chunk_indices],
                    "chunk_mapping": {
                        chunk_mapping[idx]["chunk_idx"]: idx 
                        for idx in doc_chunk_indices
                    }
                }
        
        return doc_embeddings
    
    def compute_cosine_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document embeddings"""
        return cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings).flatten()
    
    def compute_euclidean_distance(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance between query and document embeddings"""
        distances = euclidean_distances(query_embedding.reshape(1, -1), doc_embeddings).flatten()
        # Convert distances to similarities (1 / (1 + distance))
        similarities = 1 / (1 + distances)
        return similarities
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        if self.model is None:
            return 0
        return self.model.get_sentence_embedding_dimension()
    
    def clear_cache(self):
        """Clear the embeddings cache"""
        self.embeddings_cache.clear()
        logger.info("Embeddings cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files)
        
        return {
            "cache_dir": self.cache_dir,
            "cache_files": len(cache_files),
            "memory_cache_size": len(self.embeddings_cache),
            "disk_cache_size_mb": round(total_size / (1024 * 1024), 2),
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension()
        } 