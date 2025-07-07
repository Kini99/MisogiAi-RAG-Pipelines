#!/usr/bin/env python3
"""
Test script for the Intelligent Document Chunking System
Demonstrates the core functionality of the system.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from document_processor import DocumentProcessor
from document_classifier import DocumentType, ContentStructure
from adaptive_chunker import AdaptiveChunker, ChunkConfig
from langchain_integration import LangChainPipeline
from performance_monitor import PerformanceMonitor

def test_document_classification():
    """Test document classification functionality."""
    print("🔍 Testing Document Classification...")
    
    # Initialize classifier
    classifier = DocumentClassifier()
    
    # Test texts
    test_texts = {
        "technical_doc": """
# API Documentation
## Authentication
This section describes how to authenticate with our API.
### API Keys
Use your API key in the Authorization header.
```python
headers = {'Authorization': 'Bearer YOUR_KEY'}
```
        """,
        
        "support_ticket": """
Customer: Hi, I'm having trouble with the API.
Agent: Hello! Let me help you troubleshoot this issue.
Customer: I'm getting a 401 error.
Agent: Can you share the exact error message?
        """,
        
        "code_snippet": """
def authenticate_user(username, password):
    # Validate credentials
    if not username or not password:
        return False
    
    # Check against database
    user = db.users.find_one({'username': username})
    if user and verify_password(password, user['password']):
        return True
    return False
        """
    }
    
    for doc_type, text in test_texts.items():
        metadata = classifier.classify_document(text)
        print(f"  {doc_type}: {metadata.doc_type.value} (confidence: {metadata.confidence:.2f})")
    
    print("✅ Document classification test completed\n")

def test_adaptive_chunking():
    """Test adaptive chunking functionality."""
    print("✂️ Testing Adaptive Chunking...")
    
    # Initialize chunker
    config = ChunkConfig(
        chunk_size=500,
        chunk_overlap=100,
        preserve_code_blocks=True
    )
    chunker = AdaptiveChunker(config)
    
    # Test technical document
    tech_text = """
# API Authentication Guide

## Overview
This document provides comprehensive guidance on implementing secure authentication.

## API Key Authentication
The simplest form of authentication uses API keys.

```python
import requests

headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}

response = requests.get('https://api.example.com/users', headers=headers)
```

## OAuth 2.0 Authentication
For more secure applications, we support OAuth 2.0.

### Authorization Code Flow
1. Redirect users to the authorization endpoint
2. Handle the authorization callback
3. Exchange the authorization code for an access token

## Security Best Practices
- Store tokens securely
- Implement token refresh logic
- Set appropriate token expiration times
"""
    
    # Create metadata
    from document_classifier import DocumentMetadata
    metadata = DocumentMetadata(
        doc_type=DocumentType.TECHNICAL_DOC,
        structure=ContentStructure.HIERARCHICAL,
        confidence=0.95,
        language="en",
        has_code=True,
        has_tables=False,
        has_images=False,
        complexity_score=0.7
    )
    
    # Chunk the document
    chunks = chunker.chunk_document(tech_text, metadata, "test_tech_doc")
    
    print(f"  Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk.page_content)} chars, "
              f"context score: {chunk.metadata.get('context_score', 0):.2f}")
    
    print("✅ Adaptive chunking test completed\n")

def test_document_processing():
    """Test complete document processing pipeline."""
    print("📄 Testing Document Processing Pipeline...")
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Test with sample files
    data_dir = Path("data")
    if data_dir.exists():
        for file_path in data_dir.glob("*"):
            if file_path.is_file():
                print(f"  Processing {file_path.name}...")
                
                try:
                    result = processor.process_document(str(file_path))
                    
                    if result.success:
                        print(f"    ✅ Success: {result.doc_type.value}")
                        print(f"    📊 Chunks: {len(result.chunks)}")
                        print(f"    ⏱️ Time: {result.processing_time:.2f}s")
                        print(f"    🎯 Confidence: {result.metadata.confidence:.2f}")
                    else:
                        print(f"    ❌ Failed: {result.error_message}")
                        
                except Exception as e:
                    print(f"    ❌ Error: {e}")
    
    print("✅ Document processing test completed\n")

def test_langchain_integration():
    """Test LangChain integration."""
    print("🔗 Testing LangChain Integration...")
    
    try:
        # Initialize pipeline
        pipeline = LangChainPipeline()
        
        # Test vector store stats
        stats = pipeline.get_vector_store_stats()
        print(f"  Vector store stats: {stats}")
        
        # Test search (if documents are indexed)
        try:
            results = pipeline.search_documents("authentication", k=3)
            print(f"  Search results: {len(results)} documents found")
        except Exception as e:
            print(f"  Search test skipped: {e}")
        
        print("✅ LangChain integration test completed\n")
        
    except Exception as e:
        print(f"  ❌ LangChain integration test failed: {e}\n")

def test_performance_monitoring():
    """Test performance monitoring."""
    print("📊 Testing Performance Monitoring...")
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Record some test metrics
    monitor.record_search("test query", 5, 0.5)
    monitor.record_processing("technical_doc", 1.2, 3, 800, 0.85, True)
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    print(f"  Search metrics: {metrics.get('search', {})}")
    print(f"  Processing metrics: {metrics.get('processing', {})}")
    
    # Get recommendations
    recommendations = monitor.get_recommendations()
    print(f"  Recommendations: {len(recommendations)}")
    
    print("✅ Performance monitoring test completed\n")

def main():
    """Run all tests."""
    print("🚀 Intelligent Document Chunking System - Test Suite")
    print("=" * 60)
    
    # Run tests
    test_document_classification()
    test_adaptive_chunking()
    test_document_processing()
    test_langchain_integration()
    test_performance_monitoring()
    
    print("🎉 All tests completed!")
    print("\n📝 Next Steps:")
    print("1. Start the web application: python app.py")
    print("2. Open http://localhost:8000 in your browser")
    print("3. Upload documents and test the system")
    print("4. Check the API documentation at http://localhost:8000/docs")

if __name__ == "__main__":
    main() 