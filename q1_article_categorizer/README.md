# 📰 Article Classification System

A comprehensive system that automatically classifies articles into 6 categories (Tech, Finance, Healthcare, Sports, Politics, Entertainment) using 4 different embedding approaches. Built with Streamlit for the web interface and Python for the backend.

## 🎯 Features

- **4 Embedding Models**: GloVe, BERT, Sentence-BERT, and OpenAI text-embedding-ada-002
- **Real-time Classification**: Get predictions from all models simultaneously
- **Interactive Visualizations**: UMAP clustering, performance comparisons, and probability heatmaps
- **Model Comparison**: Compare accuracy, precision, recall, and F1-scores across all embedders
- **User-friendly Interface**: Clean Streamlit web app with intuitive navigation
- **Model Persistence**: Save and load trained models for future use

## 🏗️ Architecture

### Embedding Models

1. **GloVe (Global Vectors for Word Representation)**
   - Pre-trained word vectors from Stanford
   - Average pooling for document representation
   - Fast and lightweight (100-dimensional)

2. **BERT (Bidirectional Encoder Representations from Transformers)**
   - Uses [CLS] token embeddings
   - Contextual understanding of text
   - 768-dimensional embeddings

3. **Sentence-BERT (all-MiniLM-L6-v2)**
   - Optimized for sentence-level embeddings
   - Faster than BERT while maintaining performance
   - 384-dimensional embeddings

4. **OpenAI text-embedding-ada-002**
   - Latest OpenAI embedding model
   - High-dimensional (1536 dimensions)
   - Requires API key

### Classification Pipeline

- **Text Preprocessing**: Cleaning and normalization
- **Embedding Generation**: Convert text to vector representations
- **Feature Scaling**: StandardScaler for normalization
- **Classification**: Logistic Regression with cross-validation
- **Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1-Score)

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11
- OpenAI API key (for OpenAI embeddings)
- Internet connection (for downloading models)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd q1_article_categorizer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Training Models

1. Open the web application
2. In the sidebar, click "🚀 Train Models"
3. Choose whether to use sample data or full dataset
4. Wait for training to complete (this may take several minutes)
5. View training results and performance metrics

### 2. Classifying Articles

1. Navigate to the "🎯 Classification" tab
2. Enter article text in the text area
3. Or select an example from the dropdown
4. Click "🔍 Classify Article"
5. View predictions from all 4 models with confidence scores

### 3. Analyzing Performance

1. Go to the "📊 Performance" tab
2. View performance comparison charts
3. See detailed metrics table
4. Identify the best performing model

### 4. Exploring Embeddings

1. Visit the "🔍 Embeddings" tab
2. Choose dimensionality reduction method (UMAP or PCA)
3. View embedding clusters for each model
4. Compare how different models represent the data

## 📊 Model Performance

The system evaluates each embedding approach using:

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-validation**: 5-fold CV for robust evaluation

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
EMBEDDING_MODEL=text-embedding-ada-002
```

### Model Parameters

You can modify model parameters in the respective classes:

- **GloVe**: Embedding dimension (default: 100)
- **BERT**: Model name, max length, truncation
- **Sentence-BERT**: Model name
- **OpenAI**: Model name, API configuration

## 📁 Project Structure

```
q1_article_categorizer/
├── app.py                 # Main Streamlit application
├── embeddings.py          # Embedding model implementations
├── classifier.py          # Classification pipeline
├── data_utils.py          # Data loading and preprocessing
├── visualization.py       # Plotting and visualization functions
├── requirements.txt       # Python dependencies
├── env.example           # Environment variables template
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## 🛠️ Technical Details

### Dependencies

- **Streamlit**: Web interface
- **Transformers**: BERT and Sentence-BERT models
- **Sentence-Transformers**: Sentence-BERT implementation
- **OpenAI**: OpenAI API client
- **Scikit-learn**: Machine learning pipeline
- **Plotly**: Interactive visualizations
- **UMAP**: Dimensionality reduction
- **Pandas/NumPy**: Data manipulation
- **NLTK**: Text processing

### Model Training

The system uses:
- **Logistic Regression**: Multi-class classification
- **StandardScaler**: Feature normalization
- **Cross-validation**: 5-fold CV for evaluation
- **Stratified sampling**: Maintains class distribution

### Visualization Features

- **UMAP Clustering**: 2D embedding visualization
- **Performance Charts**: Bar charts comparing metrics
- **Confidence Comparison**: Model confidence scores
- **Probability Heatmaps**: Category probability distributions
- **Interactive Plots**: Hover information and zooming

### Performance Tips

- Use sample data for quick testing
- Save trained models to avoid retraining
- Close other applications to free memory
- Use smaller embedding dimensions for faster processing
