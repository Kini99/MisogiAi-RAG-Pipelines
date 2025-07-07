"""
Embedding models for article classification.
Implements Word2Vec/GloVe, BERT, Sentence-BERT, and OpenAI embeddings.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import openai
import os
from typing import List, Union
import nltk
from nltk.tokenize import word_tokenize
import requests
import zipfile
import io
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BaseEmbedder:
    """Base class for all embedding models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed text(s) into vectors."""
        raise NotImplementedError
        
    def __str__(self):
        return f"{self.__class__.__name__}({self.model_name})"


class GloVeEmbedder(BaseEmbedder):
    """GloVe word embeddings with average pooling."""
    
    def __init__(self, embedding_dim: int = 100):
        super().__init__(f"glove_{embedding_dim}d")
        self.embedding_dim = embedding_dim
        self.word_vectors = {}
        self._load_glove_vectors()
        
    def _load_glove_vectors(self):
        """Load GloVe vectors from Stanford's website."""
        glove_url = f"https://nlp.stanford.edu/data/glove.6B.zip"
        glove_file = f"glove.6B.{self.embedding_dim}d.txt"
        
        # Check if already downloaded
        if Path(glove_file).exists():
            print(f"Loading existing GloVe vectors from {glove_file}")
        else:
            print(f"Downloading GloVe vectors...")
            response = requests.get(glove_url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                zip_file.extract(glove_file)
        
        # Load vectors
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                self.word_vectors[word] = vector
                
        print(f"Loaded {len(self.word_vectors)} GloVe vectors")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts using average of word vectors."""
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        for text in texts:
            words = word_tokenize(text.lower())
            word_vecs = []
            
            for word in words:
                if word in self.word_vectors:
                    word_vecs.append(self.word_vectors[word])
            
            if word_vecs:
                # Average word vectors
                doc_embedding = np.mean(word_vecs, axis=0)
            else:
                # Zero vector if no words found
                doc_embedding = np.zeros(self.embedding_dim)
                
            embeddings.append(doc_embedding)
            
        return np.array(embeddings)


class BERTEmbedder(BaseEmbedder):
    """BERT embeddings using [CLS] token."""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts using BERT [CLS] token."""
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize and get [CLS] token
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,
                    padding=True
                )
                
                outputs = self.model(**inputs)
                # Extract [CLS] token embedding (first token)
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(cls_embedding.flatten())
                
        return np.array(embeddings)


class SentenceBERTEmbedder(BaseEmbedder):
    """Sentence-BERT embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.model = SentenceTransformer(model_name)
        
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts using Sentence-BERT."""
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding-ada-002 embeddings."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        super().__init__(model_name)
        # Load API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = openai.OpenAI(api_key=api_key)
        
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts using OpenAI API."""
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        
        for text in texts:
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error embedding text: {e}")
                # Return zero vector on error
                embeddings.append([0.0] * 1536)  # ada-002 dimension
                
        return np.array(embeddings)


def get_embedder(embedder_type: str, **kwargs):
    """Factory function to get embedder instance."""
    embedders = {
        'glove': GloVeEmbedder,
        'bert': BERTEmbedder,
        'sentence_bert': SentenceBERTEmbedder,
        'openai': OpenAIEmbedder
    }
    
    if embedder_type not in embedders:
        raise ValueError(f"Unknown embedder type: {embedder_type}")
        
    return embedders[embedder_type](**kwargs) 