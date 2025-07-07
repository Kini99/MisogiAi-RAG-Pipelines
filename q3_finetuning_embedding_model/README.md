# Fine-Tuned Embeddings for Sales Conversion Prediction

## Overview
This project fine-tunes sentence-transformer embeddings (all-MiniLM-L6-v2) on sales call transcripts to predict conversion likelihood, leveraging domain-specific patterns. It features a Streamlit app for predictions and comparison of fine-tuned vs. generic embeddings.

## Features
- Domain-specific embedding fine-tuning
- Contrastive learning for conversion prediction
- LangChain orchestration
- Streamlit interface for predictions
- Evaluation metrics: accuracy, ROC-AUC, F1, etc.

## Setup
1. **Clone the repo**
2. **Create and activate the virtual environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Run the Streamlit app:
  ```bash
  streamlit run app.py
  ```
- Upload or input call transcripts to get conversion predictions and view evaluation metrics.

## Data
- Sample call transcripts with conversion labels are provided in `data/sample_calls.csv`.

## Evaluation
- The app reports accuracy, ROC-AUC, F1, and other relevant metrics comparing fine-tuned and generic embeddings.

## Model
- Uses `sentence-transformers/all-MiniLM-L6-v2` as the base embedding model.
