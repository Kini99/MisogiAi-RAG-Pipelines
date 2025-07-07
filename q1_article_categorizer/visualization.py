"""
Visualization module for embedding clusters and performance metrics.
Uses UMAP for dimensionality reduction and Plotly for interactive plots.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
from sklearn.decomposition import PCA
from typing import Dict, List, Any
import streamlit as st


def create_embedding_clusters(embeddings: np.ndarray, labels: List[str], 
                            method: str = 'umap', n_components: int = 2) -> go.Figure:
    """
    Create 2D visualization of embeddings using UMAP or PCA.
    
    Args:
        embeddings: Array of embeddings
        labels: List of category labels
        method: 'umap' or 'pca'
        n_components: Number of components for dimensionality reduction
        
    Returns:
        Plotly figure object
    """
    if method == 'umap':
        # UMAP for dimensionality reduction
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1
        )
        coords = reducer.fit_transform(embeddings)
    else:
        # PCA for dimensionality reduction
        reducer = PCA(n_components=n_components, random_state=42)
        coords = reducer.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    df = pd.DataFrame(coords, columns=[f'Component {i+1}' for i in range(n_components)])
    df['Category'] = labels
    
    # Color mapping for categories
    colors = {
        'Tech': '#1f77b4',
        'Finance': '#ff7f0e', 
        'Healthcare': '#2ca02c',
        'Sports': '#d62728',
        'Politics': '#9467bd',
        'Entertainment': '#8c564b'
    }
    
    # Create scatter plot
    fig = px.scatter(
        df, 
        x='Component 1', 
        y='Component 2',
        color='Category',
        color_discrete_map=colors,
        title=f'Article Embeddings - {method.upper()} Visualization',
        hover_data=['Category'],
        template='plotly_white'
    )
    
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_performance_comparison(performance_df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart comparing performance metrics across embedders.
    
    Args:
        performance_df: DataFrame with performance metrics
        
    Returns:
        Plotly figure object
    """
    if performance_df.empty:
        return go.Figure()
    
    # Create subplots for different metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for metric, pos in zip(metrics, positions):
        fig.add_trace(
            go.Bar(
                x=performance_df['Embedder'],
                y=performance_df[metric],
                name=metric,
                showlegend=False,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            ),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(
        title='Model Performance Comparison',
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def create_confidence_comparison(predictions: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Create bar chart comparing confidence scores across models.
    
    Args:
        predictions: Dictionary of predictions from different models
        
    Returns:
        Plotly figure object
    """
    embedders = list(predictions.keys())
    confidences = [predictions[emb]['confidence'] for emb in embedders]
    categories = [predictions[emb]['category'] for emb in embedders]
    
    # Color mapping for categories
    colors = {
        'Tech': '#1f77b4',
        'Finance': '#ff7f0e', 
        'Healthcare': '#2ca02c',
        'Sports': '#d62728',
        'Politics': '#9467bd',
        'Entertainment': '#8c564b',
        'Unknown': '#7f7f7f'
    }
    
    bar_colors = [colors.get(cat, '#7f7f7f') for cat in categories]
    
    fig = go.Figure(data=[
        go.Bar(
            x=embedders,
            y=confidences,
            text=[f"{cat}<br>{conf:.3f}" for cat, conf in zip(categories, confidences)],
            textposition='auto',
            marker_color=bar_colors,
            hovertemplate='<b>%{x}</b><br>Category: %{text}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Model Confidence Comparison',
        xaxis_title='Embedding Model',
        yaxis_title='Confidence Score',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        height=400
    )
    
    return fig


def create_probability_heatmap(predictions: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Create heatmap showing probability distributions across categories.
    
    Args:
        predictions: Dictionary of predictions from different models
        
    Returns:
        Plotly figure object
    """
    embedders = list(predictions.keys())
    categories = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
    
    # Create probability matrix
    prob_matrix = []
    for embedder in embedders:
        probs = []
        for category in categories:
            prob = predictions[embedder]['probabilities'].get(category, 0.0)
            probs.append(prob)
        prob_matrix.append(probs)
    
    fig = go.Figure(data=go.Heatmap(
        z=prob_matrix,
        x=categories,
        y=embedders,
        colorscale='Viridis',
        text=[[f"{val:.3f}" for val in row] for row in prob_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Category Probability Distribution',
        xaxis_title='Categories',
        yaxis_title='Embedding Models',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_model_metrics_table(performance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a formatted table for display in Streamlit.
    
    Args:
        performance_df: DataFrame with performance metrics
        
    Returns:
        Formatted DataFrame for display
    """
    if performance_df.empty:
        return pd.DataFrame()
    
    # Round numeric columns
    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean', 'CV Std']
    for col in numeric_cols:
        if col in performance_df.columns:
            performance_df[col] = performance_df[col].round(3)
    
    return performance_df


def plot_embedding_evolution(embeddings_dict: Dict[str, np.ndarray], 
                           labels: List[str]) -> go.Figure:
    """
    Create side-by-side comparison of embeddings from different models.
    
    Args:
        embeddings_dict: Dictionary mapping model names to embeddings
        labels: List of category labels
        
    Returns:
        Plotly figure object
    """
    n_models = len(embeddings_dict)
    
    if n_models == 0:
        return go.Figure()
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=n_models,
        subplot_titles=list(embeddings_dict.keys()),
        specs=[[{"secondary_y": False}] * n_models]
    )
    
    colors = {
        'Tech': '#1f77b4',
        'Finance': '#ff7f0e', 
        'Healthcare': '#2ca02c',
        'Sports': '#d62728',
        'Politics': '#9467bd',
        'Entertainment': '#8c564b'
    }
    
    for i, (model_name, embeddings) in enumerate(embeddings_dict.items(), 1):
        # Reduce dimensionality
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(embeddings)
        
        # Create DataFrame
        df = pd.DataFrame(coords, columns=['Component 1', 'Component 2'])
        df['Category'] = labels
        
        # Add scatter plot
        for category in df['Category'].unique():
            mask = df['Category'] == category
            fig.add_trace(
                go.Scatter(
                    x=df[mask]['Component 1'],
                    y=df[mask]['Component 2'],
                    mode='markers',
                    name=f"{model_name} - {category}",
                    marker=dict(color=colors.get(category, '#7f7f7f')),
                    showlegend=(i == 1),  # Only show legend for first subplot
                    hovertemplate=f'<b>{category}</b><extra></extra>'
                ),
                row=1, col=i
            )
    
    fig.update_layout(
        title='Embedding Comparison Across Models',
        height=500,
        template='plotly_white'
    )
    
    return fig 