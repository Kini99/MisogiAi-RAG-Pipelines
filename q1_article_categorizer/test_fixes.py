#!/usr/bin/env python3
"""
Test script to verify the fixes for the embedding visualization error.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils import load_training_data, get_category_examples
from classifier import ArticleClassifier
from embeddings import get_embedder

def test_data_loading():
    """Test that data loading works with the fixed parameter."""
    print("Testing data loading...")
    
    try:
        # Test with sample data
        texts, labels = load_training_data(use_sample=True)
        print(f"✅ Sample data loaded: {len(texts)} texts, {len(labels)} labels")
        print(f"Categories: {set(labels)}")
        
        # Test with AG News data (if available)
        texts_ag, labels_ag = load_training_data(use_sample=False)
        print(f"✅ AG News data loaded: {len(texts_ag)} texts, {len(labels_ag)} labels")
        print(f"Categories: {set(labels_ag)}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_classifier_training():
    """Test that classifier training works."""
    print("\nTesting classifier training...")
    
    try:
        # Load sample data
        texts, labels = load_training_data(use_sample=True)
        
        # Initialize classifier
        classifier = ArticleClassifier()
        classifier.setup_embedders()
        
        print(f"Available embedders: {list(classifier.embedders.keys())}")
        
        # Train one embedder (GloVe should be available)
        if 'glove' in classifier.embedders:
            print("Training GloVe classifier...")
            metrics = classifier.train_embedder_classifier('glove', texts, labels)
            print(f"✅ GloVe training completed: {metrics}")
            
            # Test prediction
            test_text = "Apple launches new iPhone with advanced AI features"
            predictions = classifier.predict(test_text)
            print(f"✅ Prediction test: {predictions}")
            
            return True
        else:
            print("❌ No embedders available for testing")
            return False
            
    except Exception as e:
        print(f"❌ Classifier training failed: {e}")
        return False

def test_embedding_models():
    """Test that embedding models can be initialized."""
    print("\nTesting embedding models...")
    
    try:
        # Test GloVe
        glove_embedder = get_embedder('glove', embedding_dim=100)
        print("✅ GloVe embedder initialized")
        
        # Test embedding
        test_texts = ["Hello world", "Test sentence"]
        embeddings = glove_embedder.embed(test_texts)
        print(f"✅ GloVe embeddings shape: {embeddings.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Testing fixes for embedding visualization error...\n")
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Embedding Models", test_embedding_models),
        ("Classifier Training", test_classifier_training),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("📊 Test Results Summary:")
    print("=" * 40)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! The fixes should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 