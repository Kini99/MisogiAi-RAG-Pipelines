"""
Document Processor for Intelligent Chunking System
Handles different file formats and integrates classification and chunking.
"""

import os
import re
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from pypdf import PdfReader
from docx import Document as DocxDocument
import markdown
from bs4 import BeautifulSoup

from .document_classifier import DocumentClassifier, DocumentMetadata, DocumentType
from .adaptive_chunker import AdaptiveChunker, ChunkConfig
from langchain.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of document processing."""
    doc_id: str
    file_path: str
    doc_type: DocumentType
    metadata: DocumentMetadata
    chunks: List[Document]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class DocumentProcessor:
    """
    Main document processor that orchestrates classification and chunking.
    """
    
    def __init__(self, 
                 classifier_model_path: Optional[str] = None,
                 chunk_config: Optional[ChunkConfig] = None):
        self.classifier = DocumentClassifier(classifier_model_path)
        self.chunker = AdaptiveChunker(chunk_config)
        self.supported_formats = {
            '.txt': self._process_text,
            '.md': self._process_markdown,
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.html': self._process_html,
            '.csv': self._process_csv,
            '.json': self._process_json
        }
    
    def process_document(self, file_path: str, doc_id: Optional[str] = None) -> ProcessingResult:
        """
        Process a single document through classification and chunking.
        """
        start_time = datetime.now()
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ProcessingResult(
                doc_id=doc_id or file_path.stem,
                file_path=str(file_path),
                doc_type=DocumentType.UNKNOWN,
                metadata=None,
                chunks=[],
                processing_time=0.0,
                success=False,
                error_message=f"File not found: {file_path}"
            )
        
        try:
            # Extract text based on file format
            text = self._extract_text(file_path)
            if not text.strip():
                return ProcessingResult(
                    doc_id=doc_id or file_path.stem,
                    file_path=str(file_path),
                    doc_type=DocumentType.UNKNOWN,
                    metadata=None,
                    chunks=[],
                    processing_time=0.0,
                    success=False,
                    error_message="No text content extracted"
                )
            
            # Classify document
            metadata = self.classifier.classify_document(text, str(file_path))
            
            # Chunk document
            chunks = self.chunker.chunk_document(text, metadata, doc_id or file_path.stem)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                doc_id=doc_id or file_path.stem,
                file_path=str(file_path),
                doc_type=metadata.doc_type,
                metadata=metadata,
                chunks=chunks,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error processing {file_path}: {e}")
            
            return ProcessingResult(
                doc_id=doc_id or file_path.stem,
                file_path=str(file_path),
                doc_type=DocumentType.UNKNOWN,
                metadata=None,
                chunks=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def process_directory(self, directory_path: str, 
                         recursive: bool = True) -> List[ProcessingResult]:
        """
        Process all supported documents in a directory.
        """
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        results = []
        
        # Find all files
        if recursive:
            files = list(directory.rglob('*'))
        else:
            files = list(directory.glob('*'))
        
        # Filter for supported formats
        supported_files = [
            f for f in files 
            if f.is_file() and f.suffix.lower() in self.supported_formats
        ]
        
        logger.info(f"Found {len(supported_files)} supported files in {directory_path}")
        
        for file_path in supported_files:
            result = self.process_document(str(file_path))
            results.append(result)
            
            if result.success:
                logger.info(f"Successfully processed {file_path.name} "
                          f"({len(result.chunks)} chunks, {result.processing_time:.2f}s)")
            else:
                logger.warning(f"Failed to process {file_path.name}: {result.error_message}")
        
        return results
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text from file based on its format."""
        suffix = file_path.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        return self.supported_formats[suffix](file_path)
    
    def _process_text(self, file_path: Path) -> str:
        """Process plain text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _process_markdown(self, file_path: Path) -> str:
        """Process markdown files."""
        text = self._process_text(file_path)
        
        # Convert markdown to HTML for better structure analysis
        html = markdown.markdown(text)
        
        # Extract text from HTML while preserving structure
        soup = BeautifulSoup(html, 'html.parser')
        
        # Preserve headers and structure
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tag.insert_before(f"\n{tag.name[1] * '#'} {tag.get_text()}\n")
        
        # Preserve code blocks
        for tag in soup.find_all('code'):
            if tag.parent.name == 'pre':
                tag.insert_before(f"\n```\n{tag.get_text()}\n```\n")
            else:
                tag.insert_before(f"`{tag.get_text()}`")
        
        # Get clean text
        return soup.get_text()
    
    def _process_pdf(self, file_path: Path) -> str:
        """Process PDF files."""
        try:
            reader = PdfReader(file_path)
            text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
    
    def _process_docx(self, file_path: Path) -> str:
        """Process DOCX files."""
        try:
            doc = DocxDocument(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    # Preserve heading structure
                    if paragraph.style.name.startswith('Heading'):
                        level = paragraph.style.name[-1]
                        text += f"\n{'#' * int(level)} {paragraph.text}\n"
                    else:
                        text += paragraph.text + "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            raise
    
    def _process_html(self, file_path: Path) -> str:
        """Process HTML files."""
        text = self._process_text(file_path)
        
        soup = BeautifulSoup(text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Preserve structure
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(tag.name[1])
            tag.insert_before(f"\n{'#' * level} {tag.get_text()}\n")
        
        # Preserve code blocks
        for tag in soup.find_all('code'):
            if tag.parent.name == 'pre':
                tag.insert_before(f"\n```\n{tag.get_text()}\n```\n")
            else:
                tag.insert_before(f"`{tag.get_text()}`")
        
        return soup.get_text()
    
    def _process_csv(self, file_path: Path) -> str:
        """Process CSV files."""
        try:
            df = pd.read_csv(file_path)
            
            # Convert to markdown table format
            text = "# Data Table\n\n"
            text += df.to_markdown(index=False)
            
            # Add column descriptions
            text += "\n\n## Column Descriptions\n\n"
            for col in df.columns:
                text += f"- **{col}**: {df[col].dtype} column\n"
            
            return text
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {e}")
            raise
    
    def _process_json(self, file_path: Path) -> str:
        """Process JSON files."""
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to structured text
            text = "# JSON Data Structure\n\n"
            text += self._json_to_text(data, level=0)
            
            return text
        except Exception as e:
            logger.error(f"Error processing JSON {file_path}: {e}")
            raise
    
    def _json_to_text(self, data: Any, level: int = 0) -> str:
        """Convert JSON data to structured text."""
        indent = "  " * level
        
        if isinstance(data, dict):
            text = ""
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    text += f"{indent}- **{key}**:\n"
                    text += self._json_to_text(value, level + 1)
                else:
                    text += f"{indent}- **{key}**: {value}\n"
            return text
        elif isinstance(data, list):
            text = ""
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    text += f"{indent}- Item {i + 1}:\n"
                    text += self._json_to_text(item, level + 1)
                else:
                    text += f"{indent}- {item}\n"
            return text
        else:
            return f"{indent}- {data}\n"
    
    def get_processing_stats(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Generate processing statistics."""
        if not results:
            return {}
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        stats = {
            'total_documents': len(results),
            'successful_documents': len(successful),
            'failed_documents': len(failed),
            'success_rate': len(successful) / len(results) if results else 0,
            'total_chunks': sum(len(r.chunks) for r in successful),
            'avg_chunks_per_doc': sum(len(r.chunks) for r in successful) / len(successful) if successful else 0,
            'avg_processing_time': sum(r.processing_time for r in results) / len(results),
            'document_types': {},
            'errors': [r.error_message for r in failed if r.error_message]
        }
        
        # Document type distribution
        for result in successful:
            doc_type = result.doc_type.value
            stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
        
        return stats
    
    def export_results(self, results: List[ProcessingResult], 
                      output_dir: str, format: str = 'json') -> str:
        """Export processing results to file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'json':
            import json
            output_file = output_dir / f"processing_results_{timestamp}.json"
            
            export_data = []
            for result in results:
                export_data.append({
                    'doc_id': result.doc_id,
                    'file_path': result.file_path,
                    'doc_type': result.doc_type.value if result.doc_type else None,
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'chunk_count': len(result.chunks),
                    'error_message': result.error_message,
                    'metadata': {
                        'confidence': result.metadata.confidence if result.metadata else None,
                        'structure': result.metadata.structure.value if result.metadata else None,
                        'language': result.metadata.language if result.metadata else None,
                        'complexity_score': result.metadata.complexity_score if result.metadata else None
                    } if result.metadata else None
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            output_file = output_dir / f"processing_results_{timestamp}.csv"
            
            export_data = []
            for result in results:
                export_data.append({
                    'doc_id': result.doc_id,
                    'file_path': result.file_path,
                    'doc_type': result.doc_type.value if result.doc_type else None,
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'chunk_count': len(result.chunks),
                    'confidence': result.metadata.confidence if result.metadata else None,
                    'structure': result.metadata.structure.value if result.metadata else None,
                    'language': result.metadata.language if result.metadata else None,
                    'complexity_score': result.metadata.complexity_score if result.metadata else None,
                    'error_message': result.error_message
                })
            
            df = pd.DataFrame(export_data)
            df.to_csv(output_file, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return str(output_file) 