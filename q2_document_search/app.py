"""
Indian Legal Document Search System
Main Streamlit Application
"""

import streamlit as st
import os
import sys
import logging
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.legal_ner import LegalNER
from src.similarity_methods import SimilarityMethods
from src.metrics import SearchMetrics
from utils.helpers import setup_logging, create_directory_if_not_exists, validate_query

# Setup logging
setup_logging()

# Page configuration
st.set_page_config(
    page_title="Indian Legal Document Search System",
    page_icon="⚖️",
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
    .method-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-highlight {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #1f77b4;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all models and processors"""
    try:
        with st.spinner("Loading models..."):
            # Initialize components
            doc_processor = DocumentProcessor()
            embedding_manager = EmbeddingManager()
            legal_ner = LegalNER()
            similarity_methods = SimilarityMethods()
            search_metrics = SearchMetrics()
            
            st.success("Models loaded successfully!")
            return doc_processor, embedding_manager, legal_ner, similarity_methods, search_metrics
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

@st.cache_data
def load_sample_documents():
    """Load sample documents from data directory"""
    sample_docs = []
    data_dir = "data"
    
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        sample_docs.append({
                            'file_path': file_path,
                            'file_name': file,
                            'content': content,
                            'category': os.path.basename(root)
                        })
                    except Exception as e:
                        st.warning(f"Error loading {file}: {e}")
    
    return sample_docs

def process_uploaded_files(uploaded_files, doc_processor):
    """Process uploaded files"""
    processed_docs = []
    
    for uploaded_file in uploaded_files:
        try:
            # Save uploaded file temporarily
            temp_path = os.path.join("uploads", uploaded_file.name)
            create_directory_if_not_exists("uploads")
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process document
            doc_info = doc_processor.process_document(temp_path)
            if doc_info.get("processed", False):
                processed_docs.append(doc_info)
                st.success(f"Processed: {uploaded_file.name}")
            else:
                st.error(f"Failed to process: {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    
    return processed_docs

def display_search_results(results, method_name):
    """Display search results for a specific method"""
    st.subheader(f"{method_name.replace('_', ' ').title()} Results")
    
    if not results:
        st.info("No results found for this method.")
        return
    
    for i, result in enumerate(results[:5]):  # Show top 5 results
        with st.expander(f"Result {i+1} - Score: {result['similarity_score']:.4f}"):
            st.write("**Document Chunk:**")
            st.text_area(
                "Content",
                value=result['chunk_text'],
                height=150,
                key=f"{method_name}_{i}",
                disabled=True
            )
            
            # Show additional details for hybrid method
            if method_name == 'hybrid' and 'cosine_score' in result:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Cosine Score", f"{result['cosine_score']:.4f}")
                with col2:
                    st.metric("Entity Score", f"{result['entity_score']:.4f}")

def display_metrics_comparison(comparison_data):
    """Display metrics comparison across methods"""
    st.subheader("Performance Metrics Comparison")
    
    if not comparison_data:
        st.info("No metrics available for comparison.")
        return
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(comparison_data).T
    metrics_df = metrics_df.round(4)
    
    # Display metrics table
    st.dataframe(metrics_df, use_container_width=True)
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Precision comparison
        fig_precision = px.bar(
            x=list(comparison_data.keys()),
            y=[data['precision'] for data in comparison_data.values()],
            title="Precision Comparison",
            labels={'x': 'Method', 'y': 'Precision'}
        )
        st.plotly_chart(fig_precision, use_container_width=True)
    
    with col2:
        # Diversity comparison
        fig_diversity = px.bar(
            x=list(comparison_data.keys()),
            y=[data['diversity_score'] for data in comparison_data.values()],
            title="Diversity Score Comparison",
            labels={'x': 'Method', 'y': 'Diversity Score'}
        )
        st.plotly_chart(fig_diversity, use_container_width=True)
    
    # F1 Score comparison
    fig_f1 = px.bar(
        x=list(comparison_data.keys()),
        y=[data['f1_score'] for data in comparison_data.values()],
        title="F1 Score Comparison",
        labels={'x': 'Method', 'y': 'F1 Score'}
    )
    st.plotly_chart(fig_f1, use_container_width=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ Indian Legal Document Search System</h1>', unsafe_allow_html=True)
    st.markdown("Compare 4 different similarity methods for legal document retrieval")
    
    # Load models
    doc_processor, embedding_manager, legal_ner, similarity_methods, search_metrics = load_models()
    
    if not all([doc_processor, embedding_manager, legal_ner, similarity_methods, search_metrics]):
        st.error("Failed to load required models. Please check the installation.")
        return
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Document upload
    st.sidebar.subheader("📄 Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload legal documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF, Word, or text files"
    )
    
    # Load sample documents
    sample_docs = load_sample_documents()
    
    # Document processing
    all_documents = []
    
    if uploaded_files:
        with st.spinner("Processing uploaded documents..."):
            processed_uploaded = process_uploaded_files(uploaded_files, doc_processor)
            all_documents.extend(processed_uploaded)
    
    # Add sample documents
    if sample_docs:
        with st.spinner("Processing sample documents..."):
            for sample_doc in sample_docs:
                # Create a temporary file for processing
                temp_path = os.path.join("uploads", sample_doc['file_name'])
                create_directory_if_not_exists("uploads")
                
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(sample_doc['content'])
                
                doc_info = doc_processor.process_document(temp_path)
                if doc_info.get("processed", False):
                    all_documents.append(doc_info)
    
    if not all_documents:
        st.warning("No documents available. Please upload documents or check sample data.")
        return
    
    # Display document summary
    st.sidebar.subheader("📊 Document Summary")
    st.sidebar.write(f"Total Documents: {len(all_documents)}")
    
    total_chunks = sum(doc.get('total_chunks', 0) for doc in all_documents)
    st.sidebar.write(f"Total Chunks: {total_chunks}")
    
    # Search interface
    st.subheader("🔍 Search Interface")
    
    # Query input
    query = st.text_input(
        "Enter your legal query:",
        placeholder="e.g., Income tax deduction for education",
        help="Enter a legal query to search across documents"
    )
    
    # Query validation
    if query:
        validation = validate_query(query)
        if not validation['is_valid']:
            st.error("Query validation failed:")
            for error in validation['errors']:
                st.error(f"- {error}")
            return
        
        if validation['warnings']:
            for warning in validation['warnings']:
                st.warning(warning)
    
    # Search parameters
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results per method", 5, 20, 10)
    with col2:
        show_metrics = st.checkbox("Show detailed metrics", value=True)
    
    # Search button
    if st.button("🔍 Search Documents", type="primary"):
        if not query:
            st.error("Please enter a query.")
            return
        
        with st.spinner("Performing search..."):
            try:
                # Prepare document data
                all_chunks = []
                doc_chunks_mapping = {}
                
                for doc_idx, doc in enumerate(all_documents):
                    chunks = doc.get('chunks', [])
                    for chunk_idx, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        doc_chunks_mapping[len(all_chunks) - 1] = {
                            'doc_name': doc.get('file_name', f'Document {doc_idx}'),
                            'chunk_idx': chunk_idx
                        }
                
                if not all_chunks:
                    st.error("No document chunks available for search.")
                    return
                
                # Generate embeddings
                query_embedding = embedding_manager.get_query_embedding(query)
                doc_embeddings = embedding_manager.get_embeddings(all_chunks)
                
                # Extract entities
                query_entities = legal_ner.extract_entities(query)
                doc_entities_list = legal_ner.extract_entities_batch(all_chunks)
                
                # Perform search using all methods
                all_results = similarity_methods.search_all_methods(
                    query_embedding, doc_embeddings, all_chunks,
                    query_entities, doc_entities_list, top_k
                )
                
                # Calculate metrics
                comparison_data = search_metrics.compare_methods(
                    all_results, doc_embeddings=doc_embeddings
                )
                
                # Display results
                st.success("Search completed!")
                
                # Display query entities
                if query_entities:
                    st.subheader("🔍 Extracted Legal Entities")
                    entity_df = pd.DataFrame(query_entities)
                    st.dataframe(entity_df[['text', 'type']], use_container_width=True)
                
                # Display results in columns
                st.subheader("📋 Search Results")
                
                # Create 4 columns for results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    display_search_results(all_results.get('cosine', []), 'Cosine Similarity')
                
                with col2:
                    display_search_results(all_results.get('euclidean', []), 'Euclidean Distance')
                
                with col3:
                    display_search_results(all_results.get('mmr', []), 'MMR')
                
                with col4:
                    display_search_results(all_results.get('hybrid', []), 'Hybrid')
                
                # Display metrics
                if show_metrics:
                    display_metrics_comparison(comparison_data)
                
                # Performance summary
                st.subheader("📊 Performance Summary")
                summary = search_metrics.get_performance_summary(comparison_data)
                
                if summary:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Best Methods by Metric:**")
                        for metric, info in summary.get('best_methods', {}).items():
                            st.write(f"- {metric.title()}: {info['method']} ({info['score']:.4f})")
                    
                    with col2:
                        st.write("**Overall Statistics:**")
                        stats = summary.get('overall_stats', {})
                        if stats:
                            st.write(f"- Average Score: {stats['avg_score']:.4f}")
                            st.write(f"- Score Range: {stats['min_score']:.4f} - {stats['max_score']:.4f}")
                
            except Exception as e:
                st.error(f"Error during search: {e}")
                st.exception(e)
    
    # Test queries section
    st.subheader("🧪 Test Queries")
    st.write("Try these sample queries to test the system:")
    
    test_queries = [
        "Income tax deduction for education",
        "GST rate for textile products", 
        "Property registration process",
        "Court fee structure"
    ]
    
    for i, test_query in enumerate(test_queries):
        if st.button(f"Query {i+1}: {test_query}", key=f"test_query_{i}"):
            st.session_state.query = test_query
            st.rerun()
    
    # System information
    st.sidebar.subheader("ℹ️ System Info")
    st.sidebar.write("**Models:**")
    st.sidebar.write("- Sentence Transformers")
    st.sidebar.write("- Legal-BERT-NER")
    st.sidebar.write("- Document Processor")
    
    st.sidebar.write("**Methods:**")
    st.sidebar.write("- Cosine Similarity")
    st.sidebar.write("- Euclidean Distance")
    st.sidebar.write("- MMR (Diversity)")
    st.sidebar.write("- Hybrid (Entity + Semantic)")

if __name__ == "__main__":
    main() 