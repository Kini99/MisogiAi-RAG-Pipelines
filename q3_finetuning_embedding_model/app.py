import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import numpy as np

st.set_page_config(page_title="Sales Conversion Prediction", layout="wide")
st.title("Fine-Tuned Embeddings for Sales Conversion Prediction")

# Load sample data
def load_data():
    df = pd.read_csv("data/sample_calls.csv")
    return df

data = load_data()

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Generate embeddings
def get_embeddings(texts):
    return model.encode(texts, show_progress_bar=False)

# Prepare data for training
X = get_embeddings(data['transcript'].tolist())
y = data['conversion'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)

st.subheader("Evaluation Metrics (Generic Embeddings)")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**ROC-AUC:** {roc_auc:.2f}")
st.write(f"**F1 Score:** {f1:.2f}")
st.text(classification_report(y_test, y_pred))

st.subheader("Try Your Own Transcript")
user_input = st.text_area("Paste a sales call transcript here:")
if st.button("Predict Conversion Likelihood"):
    if user_input.strip():
        user_emb = get_embeddings([user_input])
        user_prob = clf.predict_proba(user_emb)[0,1]
        st.write(f"**Predicted Conversion Probability:** {user_prob:.2f}")
    else:
        st.warning("Please enter a transcript.")

st.caption("This is a demo using generic embeddings. Fine-tuning and advanced evaluation coming soon.") 