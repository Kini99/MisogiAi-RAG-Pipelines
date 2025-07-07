"""
Metrics Module
Calculates precision, recall, and diversity scores for search performance evaluation
"""

import numpy as np
from typing import List, Dict, Any, Set
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SearchMetrics:
    """Calculates various metrics for search performance evaluation"""
    
    def __init__(self, precision_top_k: int = 5, recall_threshold: float = 0.7):
        self.precision_top_k = precision_top_k
        self.recall_threshold = recall_threshold
    
    def calculate_precision(self, results: List[Dict[str, Any]], relevant_docs: Set[int] = None) -> float:
        """Calculate precision: relevant docs in top-k results"""
        if not results:
            return 0.0
        
        # If no relevant docs provided, assume all are relevant (baseline)
        if relevant_docs is None:
            return 1.0
        
        # Count relevant documents in top-k results
        top_k_results = results[:self.precision_top_k]
        relevant_count = 0
        
        for result in top_k_results:
            if result.get('chunk_index') in relevant_docs:
                relevant_count += 1
        
        precision = relevant_count / len(top_k_results) if top_k_results else 0.0
        return float(precision)
    
    def calculate_recall(self, results: List[Dict[str, Any]], relevant_docs: Set[int] = None) -> float:
        """Calculate recall: coverage of relevant documents"""
        if not results or relevant_docs is None:
            return 0.0
        
        # Count relevant documents found
        found_relevant = 0
        for result in results:
            if result.get('chunk_index') in relevant_docs:
                found_relevant += 1
        
        recall = found_relevant / len(relevant_docs) if relevant_docs else 0.0
        return float(recall)
    
    def calculate_diversity_score(self, results: List[Dict[str, Any]], doc_embeddings: np.ndarray) -> float:
        """Calculate diversity score: result variety"""
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
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall"""
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return float(f1)
    
    def calculate_ndcg(self, results: List[Dict[str, Any]], relevant_docs: Set[int] = None, k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if not results:
            return 0.0
        
        # If no relevant docs provided, use similarity scores as relevance
        if relevant_docs is None:
            relevance_scores = [result.get('similarity_score', 0.0) for result in results[:k]]
        else:
            relevance_scores = []
            for result in results[:k]:
                if result.get('chunk_index') in relevant_docs:
                    relevance_scores.append(1.0)
                else:
                    relevance_scores.append(0.0)
        
        # Calculate DCG
        dcg = 0.0
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return float(ndcg)
    
    def calculate_reciprocal_rank(self, results: List[Dict[str, Any]], relevant_docs: Set[int] = None) -> float:
        """Calculate Mean Reciprocal Rank"""
        if not results or relevant_docs is None:
            return 0.0
        
        for i, result in enumerate(results):
            if result.get('chunk_index') in relevant_docs:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def calculate_avg_precision(self, results: List[Dict[str, Any]], relevant_docs: Set[int] = None) -> float:
        """Calculate Average Precision"""
        if not results or relevant_docs is None:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, result in enumerate(results):
            if result.get('chunk_index') in relevant_docs:
                relevant_count += 1
                precision_at_k = relevant_count / (i + 1)
                precision_sum += precision_at_k
        
        avg_precision = precision_sum / len(relevant_docs) if relevant_docs else 0.0
        return float(avg_precision)
    
    def calculate_all_metrics(self, results: List[Dict[str, Any]], 
                            relevant_docs: Set[int] = None,
                            doc_embeddings: np.ndarray = None) -> Dict[str, float]:
        """Calculate all metrics for a set of results"""
        metrics = {}
        
        # Basic metrics
        metrics['precision'] = self.calculate_precision(results, relevant_docs)
        metrics['recall'] = self.calculate_recall(results, relevant_docs)
        metrics['f1_score'] = self.calculate_f1_score(metrics['precision'], metrics['recall'])
        
        # Advanced metrics
        metrics['ndcg'] = self.calculate_ndcg(results, relevant_docs)
        metrics['reciprocal_rank'] = self.calculate_reciprocal_rank(results, relevant_docs)
        metrics['avg_precision'] = self.calculate_avg_precision(results, relevant_docs)
        
        # Diversity metric (if embeddings provided)
        if doc_embeddings is not None:
            metrics['diversity_score'] = self.calculate_diversity_score(results, doc_embeddings)
        else:
            metrics['diversity_score'] = 0.0
        
        # Additional statistics
        if results:
            scores = [result.get('similarity_score', 0.0) for result in results]
            metrics['avg_similarity'] = float(np.mean(scores))
            metrics['max_similarity'] = float(np.max(scores))
            metrics['min_similarity'] = float(np.min(scores))
            metrics['std_similarity'] = float(np.std(scores))
        else:
            metrics['avg_similarity'] = 0.0
            metrics['max_similarity'] = 0.0
            metrics['min_similarity'] = 0.0
            metrics['std_similarity'] = 0.0
        
        return metrics
    
    def compare_methods(self, all_results: Dict[str, List[Dict[str, Any]]], 
                       relevant_docs: Set[int] = None,
                       doc_embeddings: np.ndarray = None) -> Dict[str, Dict[str, float]]:
        """Compare metrics across all similarity methods"""
        comparison = {}
        
        for method, results in all_results.items():
            metrics = self.calculate_all_metrics(results, relevant_docs, doc_embeddings)
            comparison[method] = metrics
        
        return comparison
    
    def get_performance_summary(self, comparison: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Get a summary of method performance"""
        if not comparison:
            return {}
        
        summary = {
            'best_methods': {},
            'method_rankings': {},
            'overall_stats': {}
        }
        
        # Find best method for each metric
        metrics_list = ['precision', 'recall', 'f1_score', 'ndcg', 'diversity_score']
        
        for metric in metrics_list:
            best_method = None
            best_score = -1
            
            for method, metrics in comparison.items():
                score = metrics.get(metric, 0.0)
                if score > best_score:
                    best_score = score
                    best_method = method
            
            if best_method:
                summary['best_methods'][metric] = {
                    'method': best_method,
                    'score': best_score
                }
        
        # Create rankings for each metric
        for metric in metrics_list:
            method_scores = []
            for method, metrics in comparison.items():
                score = metrics.get(metric, 0.0)
                method_scores.append((method, score))
            
            # Sort by score (descending)
            method_scores.sort(key=lambda x: x[1], reverse=True)
            summary['method_rankings'][metric] = method_scores
        
        # Overall statistics
        all_scores = []
        for method, metrics in comparison.items():
            for metric, score in metrics.items():
                all_scores.append(score)
        
        if all_scores:
            summary['overall_stats'] = {
                'avg_score': float(np.mean(all_scores)),
                'std_score': float(np.std(all_scores)),
                'min_score': float(np.min(all_scores)),
                'max_score': float(np.max(all_scores))
            }
        
        return summary 