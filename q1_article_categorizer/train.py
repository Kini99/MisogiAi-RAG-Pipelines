"""
Standalone training script for testing the Article Classification System.
This script can be run independently to train models and test the system.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from classifier import ArticleClassifier
from data_utils import load_training_data
from visualization import create_model_metrics_table

def main():
    """Main training function."""
    print("🚀 Article Classification System - Training Script")
    print("=" * 60)
    
    # Initialize classifier
    print("\n1. Initializing classifier...")
    classifier = ArticleClassifier()
    classifier.setup_embedders()
    
    if not classifier.embedders:
        print("❌ No embedding models could be initialized.")
        print("Please check your configuration and dependencies.")
        return
    
    print(f"✅ Initialized {len(classifier.embedders)} embedding models:")
    for name in classifier.embedders.keys():
        print(f"   - {name}")
    
    # Load training data
    print("\n2. Loading training data...")
    try:
        texts, labels = load_training_data(use_sample=True)
        print(f"✅ Loaded {len(texts)} training samples")
        
        # Show category distribution
        from collections import Counter
        category_counts = Counter(labels)
        print("Category distribution:")
        for category, count in category_counts.items():
            print(f"   - {category}: {count}")
            
    except Exception as e:
        print(f"❌ Failed to load training data: {e}")
        return
    
    # Train models
    print("\n3. Training models...")
    try:
        results = classifier.train_all_classifiers(texts, labels)
        
        # Show results
        print("\n4. Training Results:")
        print("-" * 40)
        
        for embedder_name, metrics in results.items():
            if metrics:
                print(f"\n{embedder_name.upper()}:")
                print(f"  Accuracy:  {metrics['accuracy']:.3f}")
                print(f"  Precision: {metrics['precision']:.3f}")
                print(f"  Recall:    {metrics['recall']:.3f}")
                print(f"  F1-Score:  {metrics['f1_score']:.3f}")
                print(f"  CV Mean:   {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']*2:.3f})")
            else:
                print(f"\n{embedder_name.upper()}: ❌ Training failed")
        
        # Performance summary
        print("\n5. Performance Summary:")
        print("-" * 40)
        performance_df = classifier.get_performance_summary()
        if not performance_df.empty:
            formatted_df = create_model_metrics_table(performance_df)
            print(formatted_df.to_string(index=False))
            
            # Best model
            best_model = performance_df.loc[performance_df['F1-Score'].idxmax()]
            print(f"\n🏆 Best performing model: {best_model['Embedder']} (F1-Score: {best_model['F1-Score']:.3f})")
        
        # Test prediction
        print("\n6. Testing prediction...")
        test_text = "Apple launches new iPhone with advanced AI features"
        print(f"Test text: '{test_text}'")
        
        predictions = classifier.predict(test_text)
        print("\nPredictions:")
        for embedder, pred in predictions.items():
            print(f"  {embedder}: {pred['category']} (confidence: {pred['confidence']:.3f})")
        
        # Save models
        print("\n7. Saving models...")
        classifier.save_models()
        print("✅ Models saved successfully!")
        
        print("\n🎉 Training completed successfully!")
        print("\nNext steps:")
        print("1. Run 'streamlit run app.py' to start the web interface")
        print("2. Use the web app to classify new articles")
        print("3. Explore visualizations and performance metrics")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 