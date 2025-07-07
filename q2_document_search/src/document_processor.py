"""
Document Processor Module
Handles PDF and Word document parsing, text extraction, and chunking
"""

import os
import re
import pdfplumber
from docx import Document
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document parsing and text extraction from various formats"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = {'.pdf', '.docx', '.doc', '.txt'}
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from Word document"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from file based on its extension"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep legal terms
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\-\_\+\=\&\|\$\%\#\@]', '', text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text.strip()
    
    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_endings = ['.', '!', '?', '\n\n']
                for ending in sentence_endings:
                    last_ending = text.rfind(ending, start, end)
                    if last_ending > start + self.chunk_size * 0.7:  # At least 70% of chunk size
                        end = last_ending + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and return structured data"""
        try:
            # Extract text
            raw_text = self.extract_text(file_path)
            if not raw_text:
                return {"error": "No text extracted from document"}
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            # Split into chunks
            chunks = self.split_into_chunks(cleaned_text)
            
            # Create document metadata
            doc_info = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path),
                "total_chunks": len(chunks),
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "chunks": chunks,
                "processed": True
            }
            
            logger.info(f"Processed document {file_path}: {len(chunks)} chunks created")
            return doc_info
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return {"error": str(e), "file_path": file_path, "processed": False}
    
    def process_multiple_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents"""
        results = []
        for file_path in file_paths:
            result = self.process_document(file_path)
            results.append(result)
        return results
    
    def get_document_summary(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of document statistics"""
        if not doc_info.get("processed", False):
            return {"error": "Document not processed successfully"}
        
        chunks = doc_info.get("chunks", [])
        total_words = sum(len(chunk.split()) for chunk in chunks)
        avg_chunk_length = total_words / len(chunks) if chunks else 0
        
        return {
            "file_name": doc_info.get("file_name", ""),
            "total_chunks": len(chunks),
            "total_words": total_words,
            "avg_chunk_length": round(avg_chunk_length, 2),
            "file_size_mb": round(doc_info.get("file_size", 0) / (1024 * 1024), 2)
        } 