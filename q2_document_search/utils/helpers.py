"""
Helper Utilities
Common utility functions for file operations, text processing, and other tasks
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: str = "app.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_directory_if_not_exists(directory_path: str) -> bool:
    """Create directory if it doesn't exist"""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False

def get_file_extension(file_path: str) -> str:
    """Get file extension from file path"""
    return os.path.splitext(file_path)[1].lower()

def is_supported_file_format(file_path: str, supported_formats: List[str]) -> bool:
    """Check if file format is supported"""
    file_ext = get_file_extension(file_path)
    return file_ext in supported_formats

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return 0.0

def generate_file_hash(file_path: str) -> str:
    """Generate SHA-256 hash of file"""
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash for {file_path}: {e}")
        return ""

def clean_filename(filename: str) -> str:
    """Clean filename by removing special characters"""
    # Remove or replace special characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove extra spaces and underscores
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned

def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    
    # Try to break at word boundary
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can break at a reasonable word boundary
        truncated = truncated[:last_space]
    
    return truncated + suffix

def extract_section_numbers(text: str) -> List[str]:
    """Extract section numbers from legal text"""
    # Common patterns for section numbers
    patterns = [
        r'Section\s+(\d+[A-Za-z]*)',
        r'Section\s+(\d+[A-Za-z]*\s*\(\d+\))',
        r'(\d+[A-Za-z]*)\s*of\s*the\s*Act',
        r'(\d+[A-Za-z]*)\s*of\s*this\s*Act',
        r'(\d+[A-Za-z]*)\s*of\s*Income\s*Tax\s*Act',
        r'(\d+[A-Za-z]*)\s*of\s*GST\s*Act'
    ]
    
    section_numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        section_numbers.extend(matches)
    
    return list(set(section_numbers))  # Remove duplicates

def extract_legal_citations(text: str) -> List[str]:
    """Extract legal citations from text"""
    # Common citation patterns
    patterns = [
        r'(\d+\s+\w+\s+\d+)',  # Year Court Number
        r'(\w+\s+\d+\s+\w+\s+\d+)',  # Court Year Number Court
        r'(AIR\s+\d+\s+\w+\s+\d+)',  # AIR citations
        r'(SCC\s+\d+\s+\d+)',  # SCC citations
        r'(SCALE\s+\d+\s+\d+)',  # SCALE citations
        r'(ITR\s+\d+\s+\d+)',  # ITR citations
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        citations.extend(matches)
    
    return list(set(citations))  # Remove duplicates

def normalize_legal_terms(text: str) -> str:
    """Normalize common legal terms"""
    replacements = {
        'income tax': 'Income Tax',
        'gst': 'GST',
        'vat': 'VAT',
        'court': 'Court',
        'judge': 'Judge',
        'petitioner': 'Petitioner',
        'respondent': 'Respondent',
        'appellant': 'Appellant',
        'defendant': 'Defendant',
        'plaintiff': 'Plaintiff',
        'section': 'Section',
        'act': 'Act',
        'rule': 'Rule',
        'regulation': 'Regulation',
        'order': 'Order',
        'judgment': 'Judgment',
        'appeal': 'Appeal',
        'writ': 'Writ',
        'petition': 'Petition'
    }
    
    normalized_text = text
    for old, new in replacements.items():
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(old) + r'\b'
        normalized_text = re.sub(pattern, new, normalized_text, flags=re.IGNORECASE)
    
    return normalized_text

def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """Save data to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False

def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None

def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def count_words(text: str) -> int:
    """Count words in text"""
    if not text:
        return 0
    return len(text.split())

def count_sentences(text: str) -> int:
    """Count sentences in text"""
    if not text:
        return 0
    
    # Simple sentence counting using common sentence endings
    sentence_endings = ['.', '!', '?', '\n\n']
    count = 0
    
    for ending in sentence_endings:
        count += text.count(ending)
    
    return max(count, 1)  # At least 1 sentence

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis"""
    if not text:
        return []
    
    # Remove common words and punctuation
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    # Clean text and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stop words and short words
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, freq in sorted_words[:max_keywords]]
    
    return keywords

def validate_query(query: str) -> Dict[str, Any]:
    """Validate search query"""
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'word_count': 0,
        'has_legal_terms': False
    }
    
    if not query or not query.strip():
        validation['is_valid'] = False
        validation['errors'].append("Query cannot be empty")
        return validation
    
    query = query.strip()
    validation['word_count'] = count_words(query)
    
    if validation['word_count'] < 2:
        validation['warnings'].append("Query is very short. Consider adding more terms for better results.")
    
    if validation['word_count'] > 50:
        validation['warnings'].append("Query is very long. Consider simplifying for better results.")
    
    # Check for legal terms
    legal_terms = ['section', 'act', 'court', 'judgment', 'appeal', 'tax', 'gst', 'income', 'property']
    query_lower = query.lower()
    validation['has_legal_terms'] = any(term in query_lower for term in legal_terms)
    
    if not validation['has_legal_terms']:
        validation['warnings'].append("Query doesn't contain obvious legal terms. Results may be less relevant.")
    
    return validation 