"""
Document Classifier for Intelligent Chunking System
Automatically detects document types and structure patterns for optimal chunking strategies.
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Enumeration of supported document types."""
    TECHNICAL_DOC = "technical_doc"
    API_REFERENCE = "api_reference"
    SUPPORT_TICKET = "support_ticket"
    POLICY = "policy"
    TUTORIAL = "tutorial"
    CODE_SNIPPET = "code_snippet"
    TROUBLESHOOTING = "troubleshooting"
    UNKNOWN = "unknown"


class ContentStructure(Enum):
    """Enumeration of content structure patterns."""
    HIERARCHICAL = "hierarchical"
    LINEAR = "linear"
    CODE_BLOCK = "code_block"
    STEP_BY_STEP = "step_by_step"
    TABLE_FORMAT = "table_format"
    MIXED = "mixed"


@dataclass
class DocumentMetadata:
    """Metadata extracted from documents."""
    doc_type: DocumentType
    structure: ContentStructure
    confidence: float
    language: str
    has_code: bool
    has_tables: bool
    has_images: bool
    complexity_score: float


class DocumentClassifier:
    """
    Intelligent document classifier that detects content types and structure patterns.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.classifier = None
        self.vectorizer = None
        self.feature_extractors = {
            'code_patterns': self._extract_code_patterns,
            'structure_patterns': self._extract_structure_patterns,
            'content_keywords': self._extract_content_keywords,
            'formatting_patterns': self._extract_formatting_patterns
        }
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def _extract_code_patterns(self, text: str) -> Dict[str, float]:
        """Extract code-related patterns from text."""
        patterns = {
            'code_blocks': len(re.findall(r'```[\s\S]*?```', text)),
            'inline_code': len(re.findall(r'`[^`]+`', text)),
            'function_definitions': len(re.findall(r'def\s+\w+\s*\(', text)),
            'class_definitions': len(re.findall(r'class\s+\w+', text)),
            'import_statements': len(re.findall(r'import\s+\w+', text)),
            'variable_assignments': len(re.findall(r'\w+\s*=\s*[^=]+', text)),
            'method_calls': len(re.findall(r'\w+\.\w+\s*\(', text)),
            'brackets_ratio': len(re.findall(r'[{}()\[\]]', text)) / max(len(text), 1),
            'semicolon_count': text.count(';'),
            'comment_lines': len(re.findall(r'#.*$|//.*$|/\*[\s\S]*?\*/', text, re.MULTILINE))
        }
        return patterns
    
    def _extract_structure_patterns(self, text: str) -> Dict[str, float]:
        """Extract structural patterns from text."""
        lines = text.split('\n')
        patterns = {
            'heading_count': len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE)),
            'list_items': len(re.findall(r'^[\s]*[-*+]\s+', text, re.MULTILINE)),
            'numbered_list': len(re.findall(r'^[\s]*\d+\.\s+', text, re.MULTILINE)),
            'table_rows': len(re.findall(r'^\|.*\|$', text, re.MULTILINE)),
            'step_patterns': len(re.findall(r'step\s*\d+|step\s*[a-z]', text, re.IGNORECASE)),
            'paragraph_count': len([l for l in lines if l.strip() and not l.startswith('#')]),
            'empty_lines_ratio': len([l for l in lines if not l.strip()]) / max(len(lines), 1),
            'avg_line_length': sum(len(l) for l in lines) / max(len(lines), 1),
            'max_line_length': max(len(l) for l in lines) if lines else 0
        }
        return patterns
    
    def _extract_content_keywords(self, text: str) -> Dict[str, float]:
        """Extract content-specific keywords."""
        text_lower = text.lower()
        
        # Technical documentation keywords
        tech_keywords = ['api', 'endpoint', 'authentication', 'authorization', 'database', 
                        'schema', 'configuration', 'deployment', 'integration', 'sdk']
        
        # Support ticket keywords
        support_keywords = ['error', 'issue', 'problem', 'bug', 'crash', 'failed', 
                           'ticket', 'support', 'help', 'urgent', 'priority']
        
        # Policy keywords
        policy_keywords = ['policy', 'procedure', 'guideline', 'compliance', 'regulation',
                          'requirement', 'mandatory', 'prohibited', 'allowed', 'permitted']
        
        # Tutorial keywords
        tutorial_keywords = ['tutorial', 'guide', 'how to', 'example', 'demo', 'walkthrough',
                            'getting started', 'prerequisites', 'setup', 'installation']
        
        # Code snippet keywords
        code_keywords = ['function', 'method', 'class', 'variable', 'loop', 'condition',
                        'return', 'import', 'export', 'module', 'package']
        
        keyword_scores = {
            'tech_score': sum(text_lower.count(kw) for kw in tech_keywords),
            'support_score': sum(text_lower.count(kw) for kw in support_keywords),
            'policy_score': sum(text_lower.count(kw) for kw in policy_keywords),
            'tutorial_score': sum(text_lower.count(kw) for kw in tutorial_keywords),
            'code_score': sum(text_lower.count(kw) for kw in code_keywords)
        }
        
        return keyword_scores
    
    def _extract_formatting_patterns(self, text: str) -> Dict[str, float]:
        """Extract formatting and style patterns."""
        patterns = {
            'bold_text': len(re.findall(r'\*\*[^*]+\*\*', text)),
            'italic_text': len(re.findall(r'\*[^*]+\*', text)),
            'code_inline': len(re.findall(r'`[^`]+`', text)),
            'links': len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text)),
            'images': len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', text)),
            'blockquotes': len(re.findall(r'^>\s+', text, re.MULTILINE)),
            'horizontal_rules': len(re.findall(r'^---$|^___$|^\*\*\*$', text, re.MULTILINE)),
            'emphasis_ratio': (len(re.findall(r'\*\*[^*]+\*\*', text)) + 
                             len(re.findall(r'\*[^*]+\*', text))) / max(len(text), 1)
        }
        return patterns
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive features from document text."""
        features = {}
        
        # Basic text statistics
        features.update({
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'unique_words': len(set(text.lower().split())),
            'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1)
        })
        
        # Extract patterns from each feature extractor
        for extractor_name, extractor_func in self.feature_extractors.items():
            features.update(extractor_func(text))
        
        return features
    
    def classify_document(self, text: str, file_path: Optional[str] = None) -> DocumentMetadata:
        """
        Classify a document and return metadata with confidence scores.
        """
        features = self.extract_features(text)
        
        # Rule-based classification as fallback
        if self.classifier is None:
            return self._rule_based_classification(text, features)
        
        # ML-based classification
        try:
            feature_vector = self.vectorizer.transform([str(features)])
            prediction = self.classifier.predict(feature_vector)[0]
            confidence = max(self.classifier.predict_proba(feature_vector)[0])
            
            doc_type = DocumentType(prediction)
        except Exception as e:
            logger.warning(f"ML classification failed: {e}. Falling back to rule-based.")
            return self._rule_based_classification(text, features)
        
        # Determine structure
        structure = self._determine_structure(text, features)
        
        # Additional metadata
        metadata = DocumentMetadata(
            doc_type=doc_type,
            structure=structure,
            confidence=confidence,
            language=self._detect_language(text),
            has_code=features['code_blocks'] > 0 or features['inline_code'] > 0,
            has_tables=features['table_rows'] > 0,
            has_images=features['images'] > 0,
            complexity_score=self._calculate_complexity_score(features)
        )
        
        return metadata
    
    def _rule_based_classification(self, text: str, features: Dict[str, float]) -> DocumentMetadata:
        """Rule-based classification when ML model is not available."""
        text_lower = text.lower()
        
        # Calculate scores for each document type
        scores = {
            DocumentType.TECHNICAL_DOC: features['tech_score'] + features['heading_count'] * 2,
            DocumentType.API_REFERENCE: features['api_score'] if 'api_score' in features else 0,
            DocumentType.SUPPORT_TICKET: features['support_score'] + features['step_patterns'],
            DocumentType.POLICY: features['policy_score'] + features['heading_count'],
            DocumentType.TUTORIAL: features['tutorial_score'] + features['step_patterns'] * 2,
            DocumentType.CODE_SNIPPET: features['code_blocks'] * 10 + features['inline_code'] * 2,
            DocumentType.TROUBLESHOOTING: features['support_score'] + features['step_patterns'] * 3
        }
        
        # Find the best match
        best_type = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_type[1] / 10, 0.95)  # Normalize confidence
        
        structure = self._determine_structure(text, features)
        
        return DocumentMetadata(
            doc_type=best_type[0],
            structure=structure,
            confidence=confidence,
            language=self._detect_language(text),
            has_code=features['code_blocks'] > 0 or features['inline_code'] > 0,
            has_tables=features['table_rows'] > 0,
            has_images=features['images'] > 0,
            complexity_score=self._calculate_complexity_score(features)
        )
    
    def _determine_structure(self, text: str, features: Dict[str, float]) -> ContentStructure:
        """Determine the content structure pattern."""
        if features['code_blocks'] > 0:
            return ContentStructure.CODE_BLOCK
        elif features['step_patterns'] > 2:
            return ContentStructure.STEP_BY_STEP
        elif features['table_rows'] > 3:
            return ContentStructure.TABLE_FORMAT
        elif features['heading_count'] > 3:
            return ContentStructure.HIERARCHICAL
        elif features['list_items'] > 5 or features['numbered_list'] > 5:
            return ContentStructure.LINEAR
        else:
            return ContentStructure.MIXED
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # This is a simplified version - in production, use a proper language detection library
        if re.search(r'[а-яё]', text, re.IGNORECASE):
            return 'ru'
        elif re.search(r'[一-龯]', text):
            return 'zh'
        elif re.search(r'[あ-ん]', text):
            return 'ja'
        else:
            return 'en'
    
    def _calculate_complexity_score(self, features: Dict[str, float]) -> float:
        """Calculate document complexity score."""
        complexity_factors = [
            features['text_length'] / 1000,  # Length factor
            features['heading_count'] / 10,  # Structure factor
            features['code_blocks'] / 5,     # Code complexity
            features['table_rows'] / 10,     # Data complexity
            features['unique_words'] / features.get('word_count', 1)  # Vocabulary diversity
        ]
        return sum(complexity_factors) / len(complexity_factors)
    
    def train_model(self, training_data: List[Tuple[str, DocumentType]]):
        """Train the ML classifier with labeled data."""
        texts, labels = zip(*training_data)
        
        # Extract features for all documents
        feature_vectors = []
        for text in texts:
            features = self.extract_features(text)
            feature_vectors.append(str(features))
        
        # Create and train vectorizer
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(feature_vectors)
        
        # Create and train classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X, [label.value for label in labels])
        
        logger.info("Document classifier trained successfully")
    
    def save_model(self, model_path: str):
        """Save the trained model."""
        if self.classifier and self.vectorizer:
            model_data = {
                'classifier': self.classifier,
                'vectorizer': self.vectorizer
            }
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        try:
            model_data = joblib.load(model_path)
            self.classifier = model_data['classifier']
            self.vectorizer = model_data['vectorizer']
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def evaluate_model(self, test_data: List[Tuple[str, DocumentType]]) -> Dict:
        """Evaluate model performance on test data."""
        if not self.classifier or not self.vectorizer:
            raise ValueError("Model not trained or loaded")
        
        texts, true_labels = zip(*test_data)
        
        # Extract features
        feature_vectors = []
        for text in texts:
            features = self.extract_features(text)
            feature_vectors.append(str(features))
        
        X_test = self.vectorizer.transform(feature_vectors)
        y_true = [label.value for label in true_labels]
        y_pred = self.classifier.predict(X_test)
        
        # Generate evaluation report
        report = classification_report(y_true, y_pred, output_dict=True)
        return report 