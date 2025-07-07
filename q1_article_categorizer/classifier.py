"""
Classification module for article categorization.
Trains and evaluates Logistic Regression classifiers on different embeddings.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure scikit-learn to use single-threaded processing
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from embeddings import get_embedder


class ArticleClassifier:
    """Main classifier class that handles training and prediction."""
    
    def __init__(self):
        self.categories = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
        self.category_to_id = {cat: i for i, cat in enumerate(self.categories)}
        self.id_to_category = {i: cat for i, cat in enumerate(self.categories)}
        
        self.embedders = {}
        self.classifiers = {}
        self.scalers = {}
        self.performance_metrics = {}
        
    def setup_embedders(self):
        """Initialize all embedding models."""
        print("Setting up embedding models...")
        
        # GloVe embedder
        try:
            self.embedders['glove'] = get_embedder('glove', embedding_dim=100)
            print("✓ GloVe embedder initialized")
        except Exception as e:
            print(f"✗ Failed to initialize GloVe: {e}")
            
        # BERT embedder
        try:
            # Set torch to single-threaded mode to avoid conflicts
            import torch
            torch.set_num_threads(1)
            self.embedders['bert'] = get_embedder('bert')
            print("✓ BERT embedder initialized")
        except Exception as e:
            print(f"✗ Failed to initialize BERT: {e}")
            
        # Sentence-BERT embedder
        try:
            self.embedders['sentence_bert'] = get_embedder('sentence_bert')
            print("✓ Sentence-BERT embedder initialized")
        except Exception as e:
            print(f"✗ Failed to initialize Sentence-BERT: {e}")
            
        # OpenAI embedder
        try:
            self.embedders['openai'] = get_embedder('openai')
            print("✓ OpenAI embedder initialized")
        except Exception as e:
            print(f"✗ Failed to initialize OpenAI: {e}")
    
    def prepare_data(self, texts: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        # Convert labels to numeric
        y = np.array([self.category_to_id[label] for label in labels])
        
        # Combine title and text if available
        processed_texts = []
        for text in texts:
            if isinstance(text, dict):
                # If text is a dict with 'title' and 'text' keys
                title = text.get('title', '')
                content = text.get('text', '')
                processed_text = f"{title} {content}".strip()
            else:
                processed_text = str(text)
            processed_texts.append(processed_text)
            
        return processed_texts, y
    
    def train_embedder_classifier(self, embedder_name: str, texts: List[str], 
                                labels: List[str], test_size: float = 0.2) -> Dict[str, float]:
        """Train classifier for a specific embedder."""
        print(f"\nTraining {embedder_name} classifier...")
        
        # Prepare data
        processed_texts, y = self.prepare_data(texts, labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Get embedder
        embedder = self.embedders[embedder_name]
        
        # Generate embeddings
        print(f"Generating {embedder_name} embeddings...")
        X_train_emb = embedder.embed(X_train)
        X_test_emb = embedder.embed(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_emb)
        X_test_scaled = scaler.transform(X_test_emb)
        
        # Train classifier
        classifier = LogisticRegression(
            max_iter=1000, 
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs'
        )
        
        # Dynamically determine cv folds based on class distribution
        from collections import Counter
        class_counts = Counter(y_train)
        min_class_count = min(class_counts.values())
        cv_folds = max(2, min(5, min_class_count))
        
        # Cross-validation
        cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=cv_folds)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train on full training set
        classifier.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = classifier.predict(X_test_scaled)
        y_pred_proba = classifier.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # Store models and metrics
        self.classifiers[embedder_name] = classifier
        self.scalers[embedder_name] = scaler
        self.performance_metrics[embedder_name] = metrics
        
        print(f"✓ {embedder_name} training completed")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        return metrics
    
    def train_all_classifiers(self, texts: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
        """Train classifiers for all available embedders."""
        print("Starting training for all embedders...")
        
        results = {}
        for embedder_name in self.embedders.keys():
            try:
                metrics = self.train_embedder_classifier(embedder_name, texts, labels)
                results[embedder_name] = metrics
            except Exception as e:
                print(f"✗ Failed to train {embedder_name}: {e}")
                results[embedder_name] = None
                
        return results
    
    def predict(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Predict category for a given text using all trained classifiers."""
        predictions = {}
        
        for embedder_name, classifier in self.classifiers.items():
            try:
                # Generate embedding
                embedder = self.embedders[embedder_name]
                embedding = embedder.embed(text)
                
                # Scale embedding
                scaler = self.scalers[embedder_name]
                embedding_scaled = scaler.transform(embedding.reshape(1, -1))
                
                # Predict
                prediction = classifier.predict(embedding_scaled)[0]
                probabilities = classifier.predict_proba(embedding_scaled)[0]
                
                # Get confidence (max probability)
                confidence = np.max(probabilities)
                
                predictions[embedder_name] = {
                    'category': self.id_to_category[prediction],
                    'confidence': confidence,
                    'probabilities': {
                        self.id_to_category[i]: prob 
                        for i, prob in enumerate(probabilities)
                    }
                }
                
            except Exception as e:
                print(f"Error predicting with {embedder_name}: {e}")
                predictions[embedder_name] = {
                    'category': 'Unknown',
                    'confidence': 0.0,
                    'probabilities': {cat: 0.0 for cat in self.categories}
                }
                
        return predictions
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary as a DataFrame."""
        if not self.performance_metrics:
            return pd.DataFrame()
            
        summary_data = []
        for embedder_name, metrics in self.performance_metrics.items():
            if metrics:
                summary_data.append({
                    'Embedder': embedder_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'CV Mean': metrics['cv_mean'],
                    'CV Std': metrics['cv_std']
                })
                
        return pd.DataFrame(summary_data)
    
    def save_models(self, directory: str = "models"):
        """Save trained models to disk."""
        os.makedirs(directory, exist_ok=True)
        
        for embedder_name in self.classifiers.keys():
            # Save classifier
            classifier_path = os.path.join(directory, f"{embedder_name}_classifier.joblib")
            joblib.dump(self.classifiers[embedder_name], classifier_path)
            
            # Save scaler
            scaler_path = os.path.join(directory, f"{embedder_name}_scaler.joblib")
            joblib.dump(self.scalers[embedder_name], scaler_path)
            
        # Save performance metrics
        metrics_path = os.path.join(directory, "performance_metrics.joblib")
        joblib.dump(self.performance_metrics, metrics_path)
        
        print(f"Models saved to {directory}/")
    
    def load_models(self, directory: str = "models"):
        """Load trained models from disk."""
        for embedder_name in self.embedders.keys():
            try:
                # Load classifier
                classifier_path = os.path.join(directory, f"{embedder_name}_classifier.joblib")
                if os.path.exists(classifier_path):
                    self.classifiers[embedder_name] = joblib.load(classifier_path)
                
                # Load scaler
                scaler_path = os.path.join(directory, f"{embedder_name}_scaler.joblib")
                if os.path.exists(scaler_path):
                    self.scalers[embedder_name] = joblib.load(scaler_path)
                    
            except Exception as e:
                print(f"Failed to load {embedder_name} models: {e}")
        
        # Load performance metrics
        metrics_path = os.path.join(directory, "performance_metrics.joblib")
        if os.path.exists(metrics_path):
            self.performance_metrics = joblib.load(metrics_path)
            
        print("Models loaded successfully") 