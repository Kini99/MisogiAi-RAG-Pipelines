"""
Similarity Methods Module
Implements four different similarity approaches for legal document retrieval
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from collections import defaultdict

logger = logging.getLogger(__name__)

class SimilarityMethods:
    """Implements various similarity methods for legal document retrieval"""
    
    def __init__(self, cosine_weight: float = 0.6, entity_weight: float = 0.4, mmr_lambda: float = 0.5):
        self.cosine_weight = cosine_weight
        self.entity_weight = entity_weight
        self.mmr_lambda = mmr_lambda
    
    def cosine_similarity_search(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray, 
                               doc_chunks: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """Standard cosine similarity search"""
        try:
            # Compute cosine similarities
            similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings).flatten()
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    'chunk_index': int(idx),
                    'chunk_text': doc_chunks[idx],
                    'similarity_score': float(similarities[idx]),
                    'method': 'cosine_similarity'
                })
            
            logger.info(f"Cosine similarity search completed. Top score: {similarities[top_indices[0]]:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in cosine similarity search: {e}")
            return []
    
    def euclidean_distance_search(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray,
                                doc_chunks: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """Euclidean distance-based search (converted to similarity)"""
        try:
            # Compute Euclidean distances
            distances = euclidean_distances(query_embedding.reshape(1, -1), doc_embeddings).flatten()
            
            # Convert distances to similarities (1 / (1 + distance))
            similarities = 1 / (1 + distances)
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    'chunk_index': int(idx),
                    'chunk_text': doc_chunks[idx],
                    'similarity_score': float(similarities[idx]),
                    'method': 'euclidean_distance'
                })
            
            logger.info(f"Euclidean distance search completed. Top score: {similarities[top_indices[0]]:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in euclidean distance search: {e}")
            return []
    
    def mmr_search(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray,
                  doc_chunks: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """Maximal Marginal Relevance search for diversity"""
        try:
            # Initial relevance scores (cosine similarity)
            relevance_scores = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings).flatten()
            
            # Initialize selected indices and remaining indices
            selected_indices = []
            remaining_indices = list(range(len(doc_embeddings)))
            
            # Select first document (highest relevance)
            first_idx = np.argmax(relevance_scores)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            
            # Iteratively select documents using MMR
            for _ in range(min(top_k - 1, len(remaining_indices))):
                if not remaining_indices:
                    break
                
                # Calculate MMR scores for remaining documents
                mmr_scores = []
                for idx in remaining_indices:
                    # Relevance component
                    relevance = relevance_scores[idx]
                    
                    # Diversity component (max similarity to already selected)
                    if selected_indices:
                        similarities_to_selected = cosine_similarity(
                            doc_embeddings[idx].reshape(1, -1),
                            doc_embeddings[selected_indices]
                        ).flatten()
                        max_similarity = np.max(similarities_to_selected)
                    else:
                        max_similarity = 0
                    
                    # MMR score
                    mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_similarity
                    mmr_scores.append(mmr_score)
                
                # Select document with highest MMR score
                best_idx_pos = np.argmax(mmr_scores)
                best_idx = remaining_indices[best_idx_pos]
                
                selected_indices.append(best_idx)
                remaining_indices.pop(best_idx_pos)
            
            # Create results
            results = []
            for idx in selected_indices:
                results.append({
                    'chunk_index': int(idx),
                    'chunk_text': doc_chunks[idx],
                    'similarity_score': float(relevance_scores[idx]),
                    'method': 'mmr'
                })
            
            logger.info(f"MMR search completed. Selected {len(results)} diverse documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in MMR search: {e}")
            return []
    
    def hybrid_similarity_search(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray,
                               doc_chunks: List[str], query_entities: List[Dict[str, Any]],
                               doc_entities_list: List[List[Dict[str, Any]]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Hybrid similarity: 0.6 × Cosine + 0.4 × Legal Entity Match"""
        try:
            # Cosine similarity component
            cosine_scores = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings).flatten()
            
            # Entity match component
            entity_scores = np.zeros(len(doc_chunks))
            for i, doc_entities in enumerate(doc_entities_list):
                if query_entities and doc_entities:
                    # Calculate entity overlap
                    query_entity_texts = {entity['text'].lower() for entity in query_entities}
                    doc_entity_texts = {entity['text'].lower() for entity in doc_entities}
                    
                    intersection = query_entity_texts.intersection(doc_entity_texts)
                    union = query_entity_texts.union(doc_entity_texts)
                    
                    if union:
                        entity_scores[i] = len(intersection) / len(union)
            
            # Combine scores
            hybrid_scores = (self.cosine_weight * cosine_scores + 
                           self.entity_weight * entity_scores)
            
            # Get top-k results
            top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    'chunk_index': int(idx),
                    'chunk_text': doc_chunks[idx],
                    'similarity_score': float(hybrid_scores[idx]),
                    'cosine_score': float(cosine_scores[idx]),
                    'entity_score': float(entity_scores[idx]),
                    'method': 'hybrid'
                })
            
            logger.info(f"Hybrid similarity search completed. Top score: {hybrid_scores[top_indices[0]]:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid similarity search: {e}")
            return []
    
    def search_all_methods(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray,
                          doc_chunks: List[str], query_entities: List[Dict[str, Any]] = None,
                          doc_entities_list: List[List[Dict[str, Any]]] = None, top_k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Run all similarity methods and return results"""
        results = {}
        
        # Cosine similarity
        results['cosine'] = self.cosine_similarity_search(
            query_embedding, doc_embeddings, doc_chunks, top_k
        )
        
        # Euclidean distance
        results['euclidean'] = self.euclidean_distance_search(
            query_embedding, doc_embeddings, doc_chunks, top_k
        )
        
        # MMR
        results['mmr'] = self.mmr_search(
            query_embedding, doc_embeddings, doc_chunks, top_k
        )
        
        # Hybrid (if entities are provided)
        if query_entities is not None and doc_entities_list is not None:
            results['hybrid'] = self.hybrid_similarity_search(
                query_embedding, doc_embeddings, doc_chunks, 
                query_entities, doc_entities_list, top_k
            )
        else:
            results['hybrid'] = []
        
        return results
    
    def calculate_diversity_score(self, results: List[Dict[str, Any]], doc_embeddings: np.ndarray) -> float:
        """Calculate diversity score for a set of results"""
        if len(results) < 2:
            return 1.0
        
        try:
            # Get embeddings for selected documents
            selected_indices = [result['chunk_index'] for result in results]
            selected_embeddings = doc_embeddings[selected_indices]
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(selected_embeddings)
            
            # Calculate diversity (1 - average similarity)
            # Exclude diagonal elements (self-similarity)
            mask = ~np.eye(similarities.shape[0], dtype=bool)
            avg_similarity = np.mean(similarities[mask])
            
            diversity_score = 1.0 - avg_similarity
            
            return float(diversity_score)
            
        except Exception as e:
            logger.error(f"Error calculating diversity score: {e}")
            return 0.0
    
    def get_method_comparison(self, all_results: Dict[str, List[Dict[str, Any]]], 
                            doc_embeddings: np.ndarray) -> Dict[str, Any]:
        """Compare performance of all methods"""
        comparison = {}
        
        for method, results in all_results.items():
            if not results:
                comparison[method] = {
                    'avg_score': 0.0,
                    'top_score': 0.0,
                    'diversity_score': 0.0,
                    'result_count': 0
                }
                continue
            
            # Calculate statistics
            scores = [result['similarity_score'] for result in results]
            avg_score = np.mean(scores)
            top_score = np.max(scores)
            diversity_score = self.calculate_diversity_score(results, doc_embeddings)
            
            comparison[method] = {
                'avg_score': float(avg_score),
                'top_score': float(top_score),
                'diversity_score': float(diversity_score),
                'result_count': len(results)
            }
        
        return comparison 