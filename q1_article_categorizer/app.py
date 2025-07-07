"""
Main Streamlit application for Article Classification System.
Provides web interface for training models and classifying articles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
import plotly.graph_objects as go

# Configure Numba to use TBB threading layer to avoid conflicts
import numba
numba.config.THREADING_LAYER = 'tbb'

# Load environment variables
load_dotenv()

# Import our modules
from classifier import ArticleClassifier
from data_utils import load_training_data, get_category_examples
from visualization import (
    create_embedding_clusters, create_performance_comparison,
    create_confidence_comparison, create_probability_heatmap,
    create_model_metrics_table, plot_embedding_evolution
)

# Page configuration
st.set_page_config(
    page_title="Article Classification System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .stButton > button {
        width: 100%;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📰 Article Classification System</h1>', unsafe_allow_html=True)
    st.markdown("### Compare 4 Different Embedding Approaches for News Classification")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model training section
        st.subheader("Model Training")
        use_sample_data = st.checkbox("Use Sample Data (for testing)", value=True)
        
        if st.button("🚀 Train Models", type="primary"):
            train_models(use_sample_data)
        
        # Model loading section
        st.subheader("Model Management")
        if st.button("📥 Load Saved Models"):
            load_saved_models()
        
        if st.button("💾 Save Models"):
            save_models()
        
        # Status
        st.subheader("📊 Status")
        if st.session_state.models_trained:
            st.success("✅ Models trained and ready")
        else:
            st.warning("⚠️ Models not trained")
        
        # Category examples
        st.subheader("📝 Category Examples")
        examples = get_category_examples()
        selected_category = st.selectbox("Select category:", list(examples.keys()))
        
        if selected_category:
            st.write("**Example articles:**")
            for i, example in enumerate(examples[selected_category][:3], 1):
                st.write(f"{i}. {example}")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Classification", "📊 Performance", "🔍 Embeddings", "ℹ️ About"])
    
    with tab1:
        classification_tab()
    
    with tab2:
        performance_tab()
    
    with tab3:
        embeddings_tab()
    
    with tab4:
        about_tab()


def train_models(use_sample_data: bool):
    """Train all embedding models."""
    st.session_state.training_progress = 0
    
    with st.spinner("Initializing classifier..."):
        classifier = ArticleClassifier()
        classifier.setup_embedders()
        st.session_state.classifier = classifier
    
    if not st.session_state.classifier.embedders:
        st.error("❌ No embedding models could be initialized. Please check your configuration.")
        return
    
    # Load training data
    with st.spinner("Loading training data..."):
        texts, labels = load_training_data(use_sample=use_sample_data)
    
    if not texts:
        st.error("❌ No training data available.")
        return
    
    # Training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Train each embedder
    embedders = list(st.session_state.classifier.embedders.keys())
    for i, embedder_name in enumerate(embedders):
        status_text.text(f"Training {embedder_name}...")
        
        try:
            st.session_state.classifier.train_embedder_classifier(
                embedder_name, texts, labels
            )
            progress = (i + 1) / len(embedders)
            progress_bar.progress(progress)
            
        except Exception as e:
            st.error(f"❌ Failed to train {embedder_name}: {e}")
    
    status_text.text("✅ Training completed!")
    st.session_state.models_trained = True
    
    # Show results
    st.success("🎉 All models trained successfully!")
    
    # Display performance summary
    performance_df = st.session_state.classifier.get_performance_summary()
    if not performance_df.empty:
        st.subheader("📈 Training Results")
        st.dataframe(performance_df, use_container_width=True)


def load_saved_models():
    """Load previously saved models."""
    if st.session_state.classifier is None:
        st.session_state.classifier = ArticleClassifier()
        st.session_state.classifier.setup_embedders()
    
    try:
        st.session_state.classifier.load_models()
        st.session_state.models_trained = True
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")


def save_models():
    """Save trained models."""
    if not st.session_state.models_trained:
        st.warning("⚠️ No trained models to save.")
        return
    
    try:
        st.session_state.classifier.save_models()
        st.success("✅ Models saved successfully!")
    except Exception as e:
        st.error(f"❌ Failed to save models: {e}")


def classification_tab():
    """Classification tab content."""
    st.header("🎯 Article Classification")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first using the sidebar.")
        return
    
    # Input section
    st.subheader("📝 Enter Article Text")
    
    # Text input
    article_text = st.text_area(
        "Paste your article text here:",
        height=200,
        placeholder="Enter the article text you want to classify..."
    )
    
    # Example articles
    examples = get_category_examples()
    selected_example = st.selectbox(
        "Or try an example:",
        ["Select an example..."] + [f"{cat}: {ex[:50]}..." for cat, exs in examples.items() for ex in exs[:1]]
    )
    
    if selected_example != "Select an example...":
        category, example = selected_example.split(": ", 1)
        for cat, exs in examples.items():
            if cat in category:
                article_text = exs[0]
                break
    
    # Classification button
    if st.button("🔍 Classify Article", type="primary"):
        if not article_text.strip():
            st.warning("⚠️ Please enter some text to classify.")
            return
        
        with st.spinner("Classifying article..."):
            predictions = st.session_state.classifier.predict(article_text)
        
        # Display results
        st.subheader("📊 Classification Results")
        
        # Separate successful and failed predictions
        successful_preds = {k: v for k, v in predictions.items() if v['category'] != 'Unknown' and v['confidence'] > 0.0}
        failed_preds = {k: v for k, v in predictions.items() if v['category'] == 'Unknown' or v['confidence'] == 0.0}
        
        if not successful_preds:
            st.error("❌ All models failed to classify the article. Please check model training, embeddings, and API keys.")
            return
        
        # Create columns for results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Confidence comparison chart (only for successful models)
            fig = create_confidence_comparison(successful_preds)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Individual model results
            st.write("**Model Predictions:**")
            for embedder, pred in successful_preds.items():
                confidence = pred['confidence']
                category = pred['category']
                
                # Color based on confidence
                if confidence > 0.8:
                    color = "🟢"
                elif confidence > 0.6:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(f"""
                <div class="prediction-card">
                    <strong>{embedder.replace('_', ' ').title()}</strong><br>
                    Category: <strong>{category}</strong><br>
                    Confidence: {color} {confidence:.3f}
                </div>
                """, unsafe_allow_html=True)
            # Show error for failed models
            for embedder in failed_preds:
                st.error(f"{embedder.replace('_', ' ').title()}: Failed to classify (model or embedding error)")
        
        # Probability heatmap (only for successful models)
        st.subheader("🎨 Probability Distribution")
        fig = create_probability_heatmap(successful_preds)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed probabilities (only for successful models)
        st.subheader("📋 Detailed Probabilities")
        prob_df = pd.DataFrame([
            {
                'Model': embedder.replace('_', ' ').title(),
                'Predicted Category': pred['category'],
                'Confidence': f"{pred['confidence']:.3f}",
                **{f"{cat}": f"{prob:.3f}" for cat, prob in pred['probabilities'].items()}
            }
            for embedder, pred in successful_preds.items()
        ])
        st.dataframe(prob_df, use_container_width=True)


def performance_tab():
    """Performance tab content."""
    st.header("📊 Model Performance")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first to see performance metrics.")
        return
    
    # Performance comparison
    performance_df = st.session_state.classifier.get_performance_summary()
    
    if not performance_df.empty:
        # Performance chart
        st.subheader("📈 Performance Comparison")
        fig = create_performance_comparison(performance_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance table
        st.subheader("📋 Detailed Metrics")
        formatted_df = create_model_metrics_table(performance_df)
        st.dataframe(formatted_df, use_container_width=True)
        
        # Best performing model
        best_model = performance_df.loc[performance_df['F1-Score'].idxmax()]
        st.success(f"🏆 Best performing model: **{best_model['Embedder']}** (F1-Score: {best_model['F1-Score']:.3f})")
    else:
        st.warning("⚠️ No performance metrics available.")


def embeddings_tab():
    """Embeddings visualization tab."""
    st.header("🔍 Embedding Visualizations")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first to see embedding visualizations.")
        return
    
    # Load some training data for visualization
    try:
        texts, labels = load_training_data(use_sample=True)
        
        if texts and labels:
            st.subheader("🎨 Embedding Clusters")
            
            # Method selection
            method = st.selectbox("Dimensionality reduction method:", ["umap", "pca"])
            
            # Generate embeddings for visualization
            embeddings_dict = {}
            for embedder_name, embedder in st.session_state.classifier.embedders.items():
                try:
                    embeddings = embedder.embed(texts[:100])  # Use first 100 for speed
                    embeddings_dict[embedder_name] = embeddings
                except Exception as e:
                    st.warning(f"Could not generate embeddings for {embedder_name}: {e}")
            
            if embeddings_dict:
                # Create visualization
                fig = plot_embedding_evolution(embeddings_dict, labels[:100])
                st.plotly_chart(fig, use_container_width=True)
                
                # Individual model visualizations
                st.subheader("📊 Individual Model Embeddings")
                selected_model = st.selectbox("Select model:", list(embeddings_dict.keys()))
                
                if selected_model:
                    fig = create_embedding_clusters(
                        embeddings_dict[selected_model], 
                        labels[:100], 
                        method=method
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error generating visualizations: {e}")


def about_tab():
    """About tab content."""
    st.header("ℹ️ About the System")
    
    st.markdown("""
    ### 🎯 Overview
    This Article Classification System compares 4 different embedding approaches for categorizing news articles into 6 categories:
    
    - **Tech** - Technology and science news
    - **Finance** - Business and financial news  
    - **Healthcare** - Medical and health-related news
    - **Sports** - Sports and athletics news
    - **Politics** - Political and government news
    - **Entertainment** - Entertainment and cultural news
    
    ### 🔧 Embedding Models
    
    1. **GloVe (Global Vectors for Word Representation)**
       - Uses pre-trained word vectors from Stanford
       - Averages word vectors to create document embeddings
       - Fast and lightweight approach
    
    2. **BERT (Bidirectional Encoder Representations from Transformers)**
       - Uses the [CLS] token embedding from BERT
       - Captures contextual relationships between words
       - State-of-the-art language understanding
    
    3. **Sentence-BERT (all-MiniLM-L6-v2)**
       - Optimized for sentence-level embeddings
       - Faster than BERT while maintaining good performance
       - Specifically designed for semantic similarity tasks
    
    4. **OpenAI text-embedding-ada-002**
       - Latest OpenAI embedding model
       - High-dimensional embeddings (1536 dimensions)
       - Requires API key for access
    
    ### 📊 Classification Pipeline
    
    - **Preprocessing**: Text cleaning and normalization
    - **Embedding**: Convert text to vector representations
    - **Scaling**: Standardize features using StandardScaler
    - **Classification**: Logistic Regression with cross-validation
    - **Evaluation**: Accuracy, Precision, Recall, F1-Score
    
    ### 🚀 Usage
    
    1. **Training**: Use the sidebar to train models on the dataset
    2. **Classification**: Enter article text and get predictions from all models
    3. **Comparison**: View confidence scores and probability distributions
    4. **Visualization**: Explore embedding clusters and model performance
    
    ### 🔑 Setup Requirements
    
    - Python 3.10 or 3.11
    - OpenAI API key (for OpenAI embeddings)
    - Internet connection (for downloading models and data)
    
    ### 📁 Project Structure
    
    ```
    ├── app.py              # Main Streamlit application
    ├── embeddings.py       # Embedding model implementations
    ├── classifier.py       # Classification pipeline
    ├── data_utils.py       # Data loading and preprocessing
    ├── visualization.py    # Plotting and visualization functions
    ├── requirements.txt    # Python dependencies
    └── env.example         # Environment variables template
    ```
    
    ### 🤝 Contributing
    
    This system is designed to be easily extensible. You can:
    - Add new embedding models by implementing the BaseEmbedder interface
    - Modify the classification pipeline in classifier.py
    - Add new visualization methods in visualization.py
    - Extend the web interface in app.py
    """)


if __name__ == "__main__":
    main() 