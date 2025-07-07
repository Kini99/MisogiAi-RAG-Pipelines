"""
Data utilities for loading and preparing training data.
Uses AG News dataset for training the classifiers.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import requests
import zipfile
import io
import os
from typing import List, Tuple, Dict
import re
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text


def load_ag_news_data() -> Tuple[List[str], List[str]]:
    """
    Load AG News dataset and map to our 6 categories.
    AG News has 4 categories: World, Sports, Business, Sci/Tech
    We'll map and expand these to our 6 categories.
    """
    print("Loading AG News dataset...")
    
    # AG News categories mapping to our categories
    category_mapping = {
        0: 'Politics',  # World -> Politics
        1: 'Sports',    # Sports -> Sports
        2: 'Finance',   # Business -> Finance
        3: 'Tech'       # Sci/Tech -> Tech
    }
    
    # Load AG News dataset
    from sklearn.datasets import fetch_20newsgroups
    
    # We'll use a subset of 20 newsgroups that maps well to our categories
    categories = ['rec.sport.baseball', 'rec.sport.hockey', 'sci.med', 
                  'sci.space', 'sci.crypt', 'sci.electronics', 'comp.graphics',
                  'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware', 'comp.windows.x', 'rec.autos',
                  'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
                  'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
                  'misc.forsale', 'talk.politics.misc', 'talk.politics.guns',
                  'talk.politics.mideast', 'talk.religion.misc', 'alt.atheism',
                  'soc.religion.christian', 'comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware', 'comp.windows.x', 'rec.autos',
                  'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
                  'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
    
    # Load data
    newsgroups = fetch_20newsgroups(subset='train', categories=categories, 
                                   remove=('headers', 'footers', 'quotes'))
    
    texts = []
    labels = []
    
    # Map categories to our 6 categories
    category_to_our_categories = {
        # Sports
        'rec.sport.baseball': 'Sports',
        'rec.sport.hockey': 'Sports',
        'rec.autos': 'Sports',
        'rec.motorcycles': 'Sports',
        
        # Tech
        'sci.space': 'Tech',
        'sci.crypt': 'Tech',
        'sci.electronics': 'Tech',
        'comp.graphics': 'Tech',
        'comp.os.ms-windows.misc': 'Tech',
        'comp.sys.ibm.pc.hardware': 'Tech',
        'comp.sys.mac.hardware': 'Tech',
        'comp.windows.x': 'Tech',
        
        # Politics
        'talk.politics.misc': 'Politics',
        'talk.politics.guns': 'Politics',
        'talk.politics.mideast': 'Politics',
        
        # Healthcare
        'sci.med': 'Healthcare',
        
        # Entertainment (using misc categories)
        'misc.forsale': 'Entertainment',
        'talk.religion.misc': 'Entertainment',
        'alt.atheism': 'Entertainment',
        'soc.religion.christian': 'Entertainment',
    }
    
    for i, text in enumerate(newsgroups.data):
        category = newsgroups.target_names[newsgroups.target[i]]
        if category in category_to_our_categories:
            our_category = category_to_our_categories[category]
            cleaned_text = clean_text(text)
            
            if len(cleaned_text) > 50:  # Filter out very short texts
                texts.append(cleaned_text)
                labels.append(our_category)
    
    print(f"Loaded {len(texts)} articles")
    print("Category distribution:")
    for cat in set(labels):
        count = labels.count(cat)
        print(f"  {cat}: {count}")
    
    return texts, labels


def create_sample_data() -> Tuple[List[str], List[str]]:
    """
    Create sample data for testing when AG News is not available.
    Returns a small dataset with examples for each category.
    """
    sample_data = [
        # Tech
        ("Apple launches new iPhone with advanced AI features", "Tech"),
        ("Microsoft releases Windows 12 with enhanced security", "Tech"),
        ("Google announces breakthrough in quantum computing", "Tech"),
        ("Tesla unveils new electric vehicle technology", "Tech"),
        ("SpaceX successfully launches satellite constellation", "Tech"),
        
        # Finance
        ("Federal Reserve raises interest rates by 0.25%", "Finance"),
        ("Bitcoin reaches new all-time high of $50,000", "Finance"),
        ("Wall Street stocks surge on positive earnings reports", "Finance"),
        ("Goldman Sachs reports record quarterly profits", "Finance"),
        ("European Central Bank maintains current monetary policy", "Finance"),
        
        # Healthcare
        ("New COVID-19 vaccine shows 95% effectiveness in trials", "Healthcare"),
        ("FDA approves breakthrough cancer treatment drug", "Healthcare"),
        ("WHO issues new guidelines for diabetes management", "Healthcare"),
        ("Medical breakthrough in Alzheimer's disease research", "Healthcare"),
        ("Global health organization reports decline in malaria cases", "Healthcare"),
        
        # Sports
        ("Lakers defeat Warriors in thrilling overtime game", "Sports"),
        ("Manchester United wins Premier League championship", "Sports"),
        ("Olympic athlete breaks world record in 100m sprint", "Sports"),
        ("Tennis star wins Grand Slam tournament", "Sports"),
        ("NFL team advances to Super Bowl after playoff victory", "Sports"),
        
        # Politics
        ("President announces new immigration policy reforms", "Politics"),
        ("Congress passes bipartisan infrastructure bill", "Politics"),
        ("Supreme Court rules on controversial voting rights case", "Politics"),
        ("International summit addresses climate change concerns", "Politics"),
        ("Election results show close race in key battleground state", "Politics"),
        
        # Entertainment
        ("Oscar-winning actor stars in new blockbuster movie", "Entertainment"),
        ("Popular TV series renewed for another season", "Entertainment"),
        ("Grammy-winning artist releases highly anticipated album", "Entertainment"),
        ("Broadway show breaks box office records", "Entertainment"),
        ("Famous director announces new film project", "Entertainment"),
    ]
    
    texts = [item[0] for item in sample_data]
    labels = [item[1] for item in sample_data]
    
    return texts, labels


def load_training_data(use_sample: bool = False) -> Tuple[List[str], List[str]]:
    """
    Load training data. Falls back to sample data if AG News is not available.
    
    Args:
        use_sample: If True, use sample data instead of AG News
        
    Returns:
        Tuple of (texts, labels)
    """
    if use_sample:
        print("Using sample data for training...")
        return create_sample_data()
    
    try:
        return load_ag_news_data()
    except Exception as e:
        print(f"Failed to load AG News dataset: {e}")
        print("Falling back to sample data...")
        return create_sample_data()


def get_category_examples() -> Dict[str, List[str]]:
    """Get example texts for each category."""
    return {
        'Tech': [
            "Apple launches new iPhone with advanced AI features",
            "Microsoft releases Windows 12 with enhanced security",
            "Google announces breakthrough in quantum computing"
        ],
        'Finance': [
            "Federal Reserve raises interest rates by 0.25%",
            "Bitcoin reaches new all-time high of $50,000",
            "Wall Street stocks surge on positive earnings reports"
        ],
        'Healthcare': [
            "New COVID-19 vaccine shows 95% effectiveness in trials",
            "FDA approves breakthrough cancer treatment drug",
            "WHO issues new guidelines for diabetes management"
        ],
        'Sports': [
            "Lakers defeat Warriors in thrilling overtime game",
            "Manchester United wins Premier League championship",
            "Olympic athlete breaks world record in 100m sprint"
        ],
        'Politics': [
            "President announces new immigration policy reforms",
            "Congress passes bipartisan infrastructure bill",
            "Supreme Court rules on controversial voting rights case"
        ],
        'Entertainment': [
            "Oscar-winning actor stars in new blockbuster movie",
            "Popular TV series renewed for another season",
            "Grammy-winning artist releases highly anticipated album"
        ]
    } 