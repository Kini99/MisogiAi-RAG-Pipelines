"""
Performance Monitor for Intelligent Document Chunking System
Tracks retrieval accuracy and refines strategies based on metrics.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import defaultdict, deque
import statistics

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class SearchMetrics:
    """Metrics for search operations."""
    query: str
    results_count: int
    response_time: float
    timestamp: datetime
    user_feedback: Optional[int] = None  # 1-5 scale
    relevant_results: Optional[int] = None


@dataclass
class QAMetrics:
    """Metrics for question-answering operations."""
    question: str
    answer: str
    tokens_used: int
    response_time: float
    timestamp: datetime
    user_feedback: Optional[int] = None  # 1-5 scale
    answer_quality: Optional[float] = None


@dataclass
class ProcessingMetrics:
    """Metrics for document processing."""
    doc_type: str
    processing_time: float
    chunk_count: int
    avg_chunk_size: float
    context_score: float
    timestamp: datetime
    success: bool


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval accuracy."""
    query_type: str
    precision: float
    recall: float
    f1_score: float
    response_time: float
    timestamp: datetime


class PerformanceMonitor:
    """
    Performance monitoring system for tracking and analyzing system performance.
    """
    
    def __init__(self, metrics_dir: str = "./metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for recent metrics
        self.search_metrics = deque(maxlen=1000)
        self.qa_metrics = deque(maxlen=1000)
        self.processing_metrics = deque(maxlen=1000)
        self.retrieval_metrics = deque(maxlen=1000)
        
        # Performance thresholds
        self.thresholds = {
            'search_response_time': 2.0,  # seconds
            'qa_response_time': 5.0,      # seconds
            'processing_time_per_chunk': 0.1,  # seconds
            'min_context_score': 0.7,
            'min_user_feedback': 3
        }
        
        # Load existing metrics
        self._load_existing_metrics()
    
    def record_search(self, query: str, results_count: int, 
                     response_time: Optional[float] = None) -> None:
        """Record search operation metrics."""
        if response_time is None:
            response_time = time.time()  # Placeholder
        
        metrics = SearchMetrics(
            query=query,
            results_count=results_count,
            response_time=response_time,
            timestamp=datetime.now()
        )
        
        self.search_metrics.append(metrics)
        self._save_metrics('search_metrics.json', metrics)
        
        # Check for performance issues
        self._check_search_performance(metrics)
    
    def record_qa_question(self, question: str, answer: str, tokens_used: int,
                          response_time: Optional[float] = None) -> None:
        """Record question-answering operation metrics."""
        if response_time is None:
            response_time = time.time()  # Placeholder
        
        metrics = QAMetrics(
            question=question,
            answer=answer,
            tokens_used=tokens_used,
            response_time=response_time,
            timestamp=datetime.now()
        )
        
        self.qa_metrics.append(metrics)
        self._save_metrics('qa_metrics.json', metrics)
        
        # Check for performance issues
        self._check_qa_performance(metrics)
    
    def record_processing(self, doc_type: str, processing_time: float,
                         chunk_count: int, avg_chunk_size: float,
                         context_score: float, success: bool) -> None:
        """Record document processing metrics."""
        metrics = ProcessingMetrics(
            doc_type=doc_type,
            processing_time=processing_time,
            chunk_count=chunk_count,
            avg_chunk_size=avg_chunk_size,
            context_score=context_score,
            timestamp=datetime.now(),
            success=success
        )
        
        self.processing_metrics.append(metrics)
        self._save_metrics('processing_metrics.json', metrics)
        
        # Check for performance issues
        self._check_processing_performance(metrics)
    
    def record_batch_processing(self, stats: Dict[str, Any]) -> None:
        """Record batch processing statistics."""
        # Extract relevant metrics from batch stats
        if 'document_types' in stats:
            for doc_type, count in stats['document_types'].items():
                # Calculate average metrics for this batch
                avg_processing_time = stats.get('avg_processing_time', 0)
                avg_chunks_per_doc = stats.get('avg_chunks_per_doc', 0)
                
                self.record_processing(
                    doc_type=doc_type,
                    processing_time=avg_processing_time,
                    chunk_count=avg_chunks_per_doc,
                    avg_chunk_size=1000,  # Default value
                    context_score=0.8,   # Default value
                    success=True
                )
    
    def record_retrieval_accuracy(self, query_type: str, precision: float,
                                 recall: float, f1_score: float,
                                 response_time: float) -> None:
        """Record retrieval accuracy metrics."""
        metrics = RetrievalMetrics(
            query_type=query_type,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            response_time=response_time,
            timestamp=datetime.now()
        )
        
        self.retrieval_metrics.append(metrics)
        self._save_metrics('retrieval_metrics.json', metrics)
    
    def add_user_feedback(self, operation_type: str, query: str, 
                         feedback_score: int, relevant_results: Optional[int] = None) -> None:
        """Add user feedback to existing metrics."""
        if operation_type == 'search':
            for metrics in self.search_metrics:
                if metrics.query == query:
                    metrics.user_feedback = feedback_score
                    metrics.relevant_results = relevant_results
                    break
        elif operation_type == 'qa':
            for metrics in self.qa_metrics:
                if metrics.question == query:
                    metrics.user_feedback = feedback_score
                    break
    
    def get_metrics(self, 
                   metric_type: Optional[str] = None,
                   time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get performance metrics."""
        cutoff_time = datetime.now() - time_range if time_range else None
        
        if metric_type == 'search' or metric_type is None:
            search_data = self._filter_metrics_by_time(self.search_metrics, cutoff_time)
            search_stats = self._calculate_search_stats(search_data)
        
        if metric_type == 'qa' or metric_type is None:
            qa_data = self._filter_metrics_by_time(self.qa_metrics, cutoff_time)
            qa_stats = self._calculate_qa_stats(qa_data)
        
        if metric_type == 'processing' or metric_type is None:
            processing_data = self._filter_metrics_by_time(self.processing_metrics, cutoff_time)
            processing_stats = self._calculate_processing_stats(processing_data)
        
        if metric_type == 'retrieval' or metric_type is None:
            retrieval_data = self._filter_metrics_by_time(self.retrieval_metrics, cutoff_time)
            retrieval_stats = self._calculate_retrieval_stats(retrieval_data)
        
        return {
            'search': search_stats if metric_type in ['search', None] else None,
            'qa': qa_stats if metric_type in ['qa', None] else None,
            'processing': processing_stats if metric_type in ['processing', None] else None,
            'retrieval': retrieval_stats if metric_type in ['retrieval', None] else None,
            'summary': self._generate_summary_stats() if metric_type is None else None
        }
    
    def generate_performance_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive performance report."""
        if output_path is None:
            output_path = self.metrics_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Get all metrics
        metrics = self.get_metrics()
        
        # Generate HTML report
        html_content = self._generate_html_report(metrics)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Performance report generated: {output_path}")
        return str(output_path)
    
    def generate_visualizations(self, output_dir: Optional[str] = None) -> List[str]:
        """Generate performance visualization charts."""
        if output_dir is None:
            output_dir = self.metrics_dir / "visualizations"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        charts = []
        
        # Response time trends
        charts.append(self._plot_response_time_trends(output_dir))
        
        # Document type processing performance
        charts.append(self._plot_document_type_performance(output_dir))
        
        # User feedback distribution
        charts.append(self._plot_user_feedback_distribution(output_dir))
        
        # Retrieval accuracy over time
        charts.append(self._plot_retrieval_accuracy(output_dir))
        
        # Processing efficiency
        charts.append(self._plot_processing_efficiency(output_dir))
        
        return charts
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance improvement recommendations."""
        recommendations = []
        
        # Analyze search performance
        search_stats = self._calculate_search_stats(list(self.search_metrics))
        if search_stats['avg_response_time'] > self.thresholds['search_response_time']:
            recommendations.append({
                'type': 'search_performance',
                'severity': 'medium',
                'description': 'Search response time is above threshold',
                'suggestion': 'Consider optimizing vector store or using faster embeddings'
            })
        
        # Analyze QA performance
        qa_stats = self._calculate_qa_stats(list(self.qa_metrics))
        if qa_stats['avg_response_time'] > self.thresholds['qa_response_time']:
            recommendations.append({
                'type': 'qa_performance',
                'severity': 'medium',
                'description': 'QA response time is above threshold',
                'suggestion': 'Consider using a faster LLM or optimizing prompts'
            })
        
        # Analyze processing performance
        processing_stats = self._calculate_processing_stats(list(self.processing_metrics))
        if processing_stats['avg_context_score'] < self.thresholds['min_context_score']:
            recommendations.append({
                'type': 'chunking_quality',
                'severity': 'high',
                'description': 'Context preservation score is below threshold',
                'suggestion': 'Review chunking strategies and adjust parameters'
            })
        
        # Analyze user feedback
        if qa_stats['avg_user_feedback'] < self.thresholds['min_user_feedback']:
            recommendations.append({
                'type': 'answer_quality',
                'severity': 'high',
                'description': 'User feedback is below acceptable threshold',
                'suggestion': 'Improve answer quality through better retrieval or LLM tuning'
            })
        
        return recommendations
    
    def _check_search_performance(self, metrics: SearchMetrics) -> None:
        """Check if search performance meets thresholds."""
        if metrics.response_time > self.thresholds['search_response_time']:
            logger.warning(f"Search response time ({metrics.response_time:.2f}s) exceeds threshold")
    
    def _check_qa_performance(self, metrics: QAMetrics) -> None:
        """Check if QA performance meets thresholds."""
        if metrics.response_time > self.thresholds['qa_response_time']:
            logger.warning(f"QA response time ({metrics.response_time:.2f}s) exceeds threshold")
    
    def _check_processing_performance(self, metrics: ProcessingMetrics) -> None:
        """Check if processing performance meets thresholds."""
        if metrics.context_score < self.thresholds['min_context_score']:
            logger.warning(f"Context score ({metrics.context_score:.2f}) below threshold")
    
    def _filter_metrics_by_time(self, metrics_list: List, cutoff_time: Optional[datetime]) -> List:
        """Filter metrics by time range."""
        if cutoff_time is None:
            return list(metrics_list)
        
        return [m for m in metrics_list if m.timestamp >= cutoff_time]
    
    def _calculate_search_stats(self, metrics: List[SearchMetrics]) -> Dict[str, Any]:
        """Calculate search performance statistics."""
        if not metrics:
            return {}
        
        response_times = [m.response_time for m in metrics]
        result_counts = [m.results_count for m in metrics]
        feedback_scores = [m.user_feedback for m in metrics if m.user_feedback is not None]
        
        return {
            'total_searches': len(metrics),
            'avg_response_time': statistics.mean(response_times),
            'max_response_time': max(response_times),
            'min_response_time': min(response_times),
            'avg_results_count': statistics.mean(result_counts),
            'avg_user_feedback': statistics.mean(feedback_scores) if feedback_scores else None,
            'recent_searches': len([m for m in metrics if m.timestamp > datetime.now() - timedelta(hours=1)])
        }
    
    def _calculate_qa_stats(self, metrics: List[QAMetrics]) -> Dict[str, Any]:
        """Calculate QA performance statistics."""
        if not metrics:
            return {}
        
        response_times = [m.response_time for m in metrics]
        token_counts = [m.tokens_used for m in metrics]
        feedback_scores = [m.user_feedback for m in metrics if m.user_feedback is not None]
        
        return {
            'total_questions': len(metrics),
            'avg_response_time': statistics.mean(response_times),
            'avg_tokens_used': statistics.mean(token_counts),
            'total_tokens_used': sum(token_counts),
            'avg_user_feedback': statistics.mean(feedback_scores) if feedback_scores else None,
            'recent_questions': len([m for m in metrics if m.timestamp > datetime.now() - timedelta(hours=1)])
        }
    
    def _calculate_processing_stats(self, metrics: List[ProcessingMetrics]) -> Dict[str, Any]:
        """Calculate processing performance statistics."""
        if not metrics:
            return {}
        
        processing_times = [m.processing_time for m in metrics]
        chunk_counts = [m.chunk_count for m in metrics]
        context_scores = [m.context_score for m in metrics]
        success_rate = sum(1 for m in metrics if m.success) / len(metrics)
        
        # Group by document type
        doc_type_stats = defaultdict(list)
        for m in metrics:
            doc_type_stats[m.doc_type].append(m)
        
        return {
            'total_documents': len(metrics),
            'success_rate': success_rate,
            'avg_processing_time': statistics.mean(processing_times),
            'avg_chunk_count': statistics.mean(chunk_counts),
            'avg_context_score': statistics.mean(context_scores),
            'document_types': {
                doc_type: {
                    'count': len(docs),
                    'avg_processing_time': statistics.mean([d.processing_time for d in docs]),
                    'avg_chunk_count': statistics.mean([d.chunk_count for d in docs]),
                    'avg_context_score': statistics.mean([d.context_score for d in docs])
                }
                for doc_type, docs in doc_type_stats.items()
            }
        }
    
    def _calculate_retrieval_stats(self, metrics: List[RetrievalMetrics]) -> Dict[str, Any]:
        """Calculate retrieval accuracy statistics."""
        if not metrics:
            return {}
        
        precisions = [m.precision for m in metrics]
        recalls = [m.recall for m in metrics]
        f1_scores = [m.f1_score for m in metrics]
        response_times = [m.response_time for m in metrics]
        
        return {
            'total_evaluations': len(metrics),
            'avg_precision': statistics.mean(precisions),
            'avg_recall': statistics.mean(recalls),
            'avg_f1_score': statistics.mean(f1_scores),
            'avg_response_time': statistics.mean(response_times)
        }
    
    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            'total_operations': len(self.search_metrics) + len(self.qa_metrics),
            'system_uptime': '24h',  # Placeholder
            'performance_grade': self._calculate_performance_grade(),
            'recommendations_count': len(self.get_recommendations())
        }
    
    def _calculate_performance_grade(self) -> str:
        """Calculate overall performance grade."""
        # Simple grading based on key metrics
        search_stats = self._calculate_search_stats(list(self.search_metrics))
        qa_stats = self._calculate_qa_stats(list(self.qa_metrics))
        
        score = 0
        
        if search_stats.get('avg_response_time', 0) < self.thresholds['search_response_time']:
            score += 25
        if qa_stats.get('avg_response_time', 0) < self.thresholds['qa_response_time']:
            score += 25
        if search_stats.get('avg_user_feedback', 0) >= self.thresholds['min_user_feedback']:
            score += 25
        if qa_stats.get('avg_user_feedback', 0) >= self.thresholds['min_user_feedback']:
            score += 25
        
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _save_metrics(self, filename: str, metrics: Any) -> None:
        """Save metrics to file."""
        filepath = self.metrics_dir / filename
        
        # Load existing metrics
        existing_metrics = []
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_metrics = json.load(f)
            except json.JSONDecodeError:
                existing_metrics = []
        
        # Add new metrics
        existing_metrics.append(asdict(metrics))
        
        # Save back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_metrics, f, indent=2, default=str)
    
    def _load_existing_metrics(self) -> None:
        """Load existing metrics from files."""
        # This is a simplified implementation
        # In production, you might want to load metrics from database
        pass
    
    def _generate_html_report(self, metrics: Dict[str, Any]) -> str:
        """Generate HTML performance report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
                .metric-label { color: #666; margin-bottom: 5px; }
                .summary { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Intelligent Document Chunking System - Performance Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Performance Grade: <span class="metric-value">{grade}</span></p>
                <p>Total Operations: {total_ops}</p>
                <p>Recommendations: {rec_count}</p>
            </div>
            
            <div class="metric-card">
                <h3>Search Performance</h3>
                <div class="metric-label">Average Response Time</div>
                <div class="metric-value">{search_time:.2f}s</div>
                <div class="metric-label">Total Searches</div>
                <div class="metric-value">{search_count}</div>
            </div>
            
            <div class="metric-card">
                <h3>QA Performance</h3>
                <div class="metric-label">Average Response Time</div>
                <div class="metric-value">{qa_time:.2f}s</div>
                <div class="metric-label">Total Questions</div>
                <div class="metric-value">{qa_count}</div>
            </div>
            
            <div class="metric-card">
                <h3>Processing Performance</h3>
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{success_rate:.1%}</div>
                <div class="metric-label">Average Context Score</div>
                <div class="metric-value">{context_score:.2f}</div>
            </div>
        </body>
        </html>
        """.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            grade=metrics.get('summary', {}).get('performance_grade', 'N/A'),
            total_ops=metrics.get('summary', {}).get('total_operations', 0),
            rec_count=metrics.get('summary', {}).get('recommendations_count', 0),
            search_time=metrics.get('search', {}).get('avg_response_time', 0),
            search_count=metrics.get('search', {}).get('total_searches', 0),
            qa_time=metrics.get('qa', {}).get('avg_response_time', 0),
            qa_count=metrics.get('qa', {}).get('total_questions', 0),
            success_rate=metrics.get('processing', {}).get('success_rate', 0),
            context_score=metrics.get('processing', {}).get('avg_context_score', 0)
        )
        
        return html
    
    def _plot_response_time_trends(self, output_dir: Path) -> str:
        """Plot response time trends."""
        # Implementation for response time visualization
        return str(output_dir / "response_time_trends.png")
    
    def _plot_document_type_performance(self, output_dir: Path) -> str:
        """Plot document type processing performance."""
        # Implementation for document type performance visualization
        return str(output_dir / "document_type_performance.png")
    
    def _plot_user_feedback_distribution(self, output_dir: Path) -> str:
        """Plot user feedback distribution."""
        # Implementation for user feedback visualization
        return str(output_dir / "user_feedback_distribution.png")
    
    def _plot_retrieval_accuracy(self, output_dir: Path) -> str:
        """Plot retrieval accuracy over time."""
        # Implementation for retrieval accuracy visualization
        return str(output_dir / "retrieval_accuracy.png")
    
    def _plot_processing_efficiency(self, output_dir: Path) -> str:
        """Plot processing efficiency metrics."""
        # Implementation for processing efficiency visualization
        return str(output_dir / "processing_efficiency.png") 