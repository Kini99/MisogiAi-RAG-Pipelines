# Indian Legal Document Search System

A comprehensive search system for Indian legal documents that compares 4 different similarity methods to find the most effective approach for legal document retrieval.

## Features

- **4 Similarity Methods**: Cosine Similarity, Euclidean Distance, MMR (Maximal Marginal Relevance), and Hybrid Similarity
- **Legal Entity Recognition**: Using available NER models for entity extraction from legal documents
- **Document Support**: PDF and Word document upload and parsing
- **Interactive UI**: Streamlit-based interface with side-by-side comparison
- **Performance Metrics**: Live precision, recall, and diversity scoring
- **Sample Dataset**: Indian Income Tax Act, GST Act, Court Judgments, and Property Law documents

## Similarity Methods

1. **Cosine Similarity**: Standard semantic matching using sentence embeddings
2. **Euclidean Distance**: Geometric distance in embedding space
3. **MMR (Maximal Marginal Relevance)**: Balances relevance and diversity to reduce redundancy
4. **Hybrid Similarity**: 0.6 × Cosine Similarity + 0.4 × Legal Entity Match

## Setup Instructions

### Prerequisites
- Python 3.11
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd q2_document_search
   ```

2. **Create and activate virtual environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
q2_document_search/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore file
├── README.md                      # This file
├── data/                          # Sample legal documents
│   ├── income_tax/
│   ├── gst/
│   ├── court_judgments/
│   └── property_law/
├── src/                           # Source code
│   ├── __init__.py
│   ├── document_processor.py      # Document parsing and preprocessing
│   ├── similarity_methods.py      # Implementation of similarity methods
│   ├── legal_ner.py              # Legal entity recognition
│   ├── embeddings.py             # Text embedding generation
│   └── metrics.py                # Performance metrics calculation
└── utils/                         # Utility functions
    ├── __init__.py
    └── helpers.py
```

## Usage

1. **Upload Documents**: Use the file uploader to add PDF or Word documents
2. **Enter Query**: Type your legal query in the search box
3. **View Results**: Compare results from all 4 similarity methods side-by-side
4. **Analyze Metrics**: Check precision, recall, and diversity scores

## Test Queries

- "Income tax deduction for education"
- "GST rate for textile products"
- "Property registration process"
- "Court fee structure"

## Performance Metrics

- **Precision**: Relevant documents in top 5 results
- **Recall**: Coverage of relevant documents
- **Diversity Score**: Result variety (especially for MMR evaluation)
