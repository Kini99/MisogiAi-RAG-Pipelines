"""
Legal NER Module
Uses available NER models for extracting entities from legal documents
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Any, Set, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class LegalNER:
    """Legal Named Entity Recognition using available NER models"""
    
    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.entity_labels = {}
        self.model_loaded = False
        
        # Load the model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the NER model and tokenizer with fallback options"""
        fallback_models = [
            "dbmdz/bert-large-cased-finetuned-conll03-english",
            "dslim/bert-base-NER",
            "Jean-Baptiste/roberta-large-ner-english"
        ]
        
        for model in [self.model_name] + [m for m in fallback_models if m != self.model_name]:
            try:
                logger.info(f"Attempting to load NER model: {model}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model)
                
                # Load model
                self.model = AutoModelForTokenClassification.from_pretrained(model)
                
                # Get entity labels
                self.entity_labels = self.model.config.id2label
                self.model_name = model
                self.model_loaded = True
                
                logger.info(f"NER model loaded successfully: {model}")
                logger.info(f"Available entity labels: {list(self.entity_labels.values())}")
                break
                
            except Exception as e:
                logger.warning(f"Failed to load model {model}: {e}")
                continue
        
        if not self.model_loaded:
            logger.error("Failed to load any NER model. Entity extraction will be disabled.")
            # Initialize with empty labels to prevent errors
            self.entity_labels = {}
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        if not text.strip() or not self.model_loaded:
            return []
        
        try:
            # Tokenize the text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # Convert predictions to tokens and labels
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            labels = [self.entity_labels[pred.item()] for pred in predictions[0]]
            
            # Extract entities
            entities = self._extract_entities_from_tokens(tokens, labels, text)
            
            logger.info(f"Extracted {len(entities)} entities from text")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_entities_from_tokens(self, tokens: List[str], labels: List[str], original_text: str) -> List[Dict[str, Any]]:
        """Extract entities from tokenized predictions"""
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            # Clean token
            clean_token = token.replace('##', '')
            
            if label.startswith('B-'):  # Beginning of entity
                # Save previous entity if exists
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity
                entity_type = label[2:]  # Remove 'B-' prefix
                current_entity = {
                    'text': clean_token,
                    'type': entity_type,
                    'start': i,
                    'end': i,
                    'confidence': 1.0
                }
                
            elif label.startswith('I-'):  # Inside entity
                entity_type = label[2:]  # Remove 'I-' prefix
                
                if current_entity and entity_type == current_entity['type']:
                    # Continue current entity
                    current_entity['text'] += ' ' + clean_token
                    current_entity['end'] = i
                else:
                    # Different entity type or no current entity
                    if current_entity:
                        entities.append(current_entity)
                    
                    # Start new entity (treat I- as B- if no current entity)
                    current_entity = {
                        'text': clean_token,
                        'type': entity_type,
                        'start': i,
                        'end': i,
                        'confidence': 1.0
                    }
            
            elif label == 'O' and current_entity:  # Outside entity
                # Save current entity
                entities.append(current_entity)
                current_entity = None
        
        # Save last entity if exists
        if current_entity:
            entities.append(current_entity)
        
        # Clean up entities
        cleaned_entities = []
        for entity in entities:
            # Remove leading/trailing whitespace
            entity['text'] = entity['text'].strip()
            
            # Filter out very short entities (likely noise)
            if len(entity['text']) > 2:
                cleaned_entities.append(entity)
        
        return cleaned_entities
    
    def get_entity_types(self) -> Set[str]:
        """Get all available entity types"""
        return set(self.entity_labels.values())
    
    def extract_entities_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract entities from multiple texts"""
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results
    
    def get_entity_overlap(self, query_entities: List[Dict[str, Any]], doc_entities: List[Dict[str, Any]]) -> float:
        """Calculate entity overlap between query and document"""
        if not query_entities or not doc_entities:
            return 0.0
        
        # Extract entity texts
        query_entity_texts = {entity['text'].lower() for entity in query_entities}
        doc_entity_texts = {entity['text'].lower() for entity in doc_entities}
        
        # Calculate overlap
        intersection = query_entity_texts.intersection(doc_entity_texts)
        union = query_entity_texts.union(doc_entity_texts)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def get_entity_similarity_score(self, query: str, document_chunk: str) -> float:
        """Calculate entity-based similarity score between query and document chunk"""
        # Extract entities from both query and document
        query_entities = self.extract_entities(query)
        doc_entities = self.extract_entities(document_chunk)
        
        # Calculate overlap
        overlap_score = self.get_entity_overlap(query_entities, doc_entities)
        
        # Additional scoring based on entity types
        type_similarity = self._calculate_entity_type_similarity(query_entities, doc_entities)
        
        # Combine scores (weighted average)
        final_score = 0.7 * overlap_score + 0.3 * type_similarity
        
        return final_score
    
    def _calculate_entity_type_similarity(self, query_entities: List[Dict[str, Any]], doc_entities: List[Dict[str, Any]]) -> float:
        """Calculate similarity based on entity types"""
        if not query_entities or not doc_entities:
            return 0.0
        
        # Get entity types
        query_types = {entity['type'] for entity in query_entities}
        doc_types = {entity['type'] for entity in doc_entities}
        
        # Calculate Jaccard similarity
        intersection = query_types.intersection(doc_types)
        union = query_types.union(doc_types)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def get_legal_keywords(self, text: str) -> List[str]:
        """Extract legal keywords from text"""
        # Define legal keywords
        legal_keywords = [
            'court', 'judgment', 'petitioner', 'respondent', 'plaintiff', 'defendant',
            'appeal', 'writ', 'petition', 'order', 'decree', 'act', 'section',
            'clause', 'regulation', 'statute', 'law', 'legal', 'jurisdiction',
            'evidence', 'witness', 'testimony', 'hearing', 'trial', 'verdict',
            'sentence', 'conviction', 'acquittal', 'bail', 'remand', 'custody'
        ]
        
        # Extract keywords from text
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in legal_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def analyze_legal_document(self, text: str) -> Dict[str, Any]:
        """Comprehensive analysis of legal document"""
        analysis = {
            'entities': self.extract_entities(text),
            'keywords': self.get_legal_keywords(text),
            'entity_types': list(self.get_entity_types()),
            'model_loaded': self.model_loaded,
            'model_name': self.model_name
        }
        
        return analysis 