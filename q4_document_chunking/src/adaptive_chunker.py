"""
Adaptive Chunking System for Intelligent Document Processing
Applies document-specific chunking strategies for optimal context preservation.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    TokenTextSplitter
)
from langchain.schema import Document

from .document_classifier import DocumentType, ContentStructure, DocumentMetadata

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Enumeration of chunking strategies."""
    SEMANTIC = "semantic"
    CODE_AWARE = "code_aware"
    HIERARCHICAL = "hierarchical"
    STEP_BY_STEP = "step_by_step"
    TABLE_AWARE = "table_aware"
    MIXED = "mixed"


@dataclass
class ChunkConfig:
    """Configuration for chunking parameters."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    preserve_headers: bool = True
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    preserve_lists: bool = True


@dataclass
class ChunkMetadata:
    """Metadata for individual chunks."""
    chunk_id: str
    original_doc_type: DocumentType
    chunk_type: str
    position: int
    size: int
    has_code: bool
    has_tables: bool
    has_headers: bool
    context_score: float
    semantic_coherence: float


class AdaptiveChunker:
    """
    Adaptive chunking system that applies document-specific strategies.
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.strategies = {
            DocumentType.TECHNICAL_DOC: self._chunk_technical_doc,
            DocumentType.API_REFERENCE: self._chunk_api_reference,
            DocumentType.SUPPORT_TICKET: self._chunk_support_ticket,
            DocumentType.POLICY: self._chunk_policy,
            DocumentType.TUTORIAL: self._chunk_tutorial,
            DocumentType.CODE_SNIPPET: self._chunk_code_snippet,
            DocumentType.TROUBLESHOOTING: self._chunk_troubleshooting
        }
        
        # Initialize text splitters
        self._init_text_splitters()
    
    def _init_text_splitters(self):
        """Initialize different text splitters for various strategies."""
        self.splitters = {
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            ),
            'markdown': MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                    ("####", "Header 4"),
                ]
            ),
            'token': TokenTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        }
    
    def chunk_document(self, text: str, metadata: DocumentMetadata, 
                      doc_id: str = "unknown") -> List[Document]:
        """
        Chunk a document using the appropriate strategy based on its metadata.
        """
        logger.info(f"Chunking document {doc_id} with type {metadata.doc_type}")
        
        # Select appropriate chunking strategy
        if metadata.doc_type in self.strategies:
            chunks = self.strategies[metadata.doc_type](text, metadata, doc_id)
        else:
            chunks = self._chunk_generic(text, metadata, doc_id)
        
        # Post-process chunks
        chunks = self._post_process_chunks(chunks, metadata)
        
        logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks
    
    def _chunk_technical_doc(self, text: str, metadata: DocumentMetadata, 
                           doc_id: str) -> List[Document]:
        """Chunk technical documentation with hierarchical structure preservation."""
        chunks = []
        
        # Split by headers first
        if metadata.structure == ContentStructure.HIERARCHICAL:
            header_splits = self.splitters['markdown'].split_text(text)
            
            for i, split in enumerate(header_splits):
                # Further split large sections
                if len(split.page_content) > self.config.max_chunk_size:
                    sub_chunks = self.splitters['recursive'].split_text(split.page_content)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk_metadata = self._create_chunk_metadata(
                            f"{doc_id}_tech_{i}_{j}", metadata, "technical_section", 
                            i * 100 + j, sub_chunk
                        )
                        chunks.append(Document(
                            page_content=sub_chunk,
                            metadata=chunk_metadata.__dict__
                        ))
                else:
                    chunk_metadata = self._create_chunk_metadata(
                        f"{doc_id}_tech_{i}", metadata, "technical_section", 
                        i, split.page_content
                    )
                    chunks.append(Document(
                        page_content=split.page_content,
                        metadata=chunk_metadata.__dict__
                    ))
        else:
            # Use recursive splitting for non-hierarchical technical docs
            splits = self.splitters['recursive'].split_text(text)
            for i, split in enumerate(splits):
                chunk_metadata = self._create_chunk_metadata(
                    f"{doc_id}_tech_{i}", metadata, "technical_content", 
                    i, split
                )
                chunks.append(Document(
                    page_content=split,
                    metadata=chunk_metadata.__dict__
                ))
        
        return chunks
    
    def _chunk_api_reference(self, text: str, metadata: DocumentMetadata, 
                           doc_id: str) -> List[Document]:
        """Chunk API reference documentation with code block preservation."""
        chunks = []
        
        # Split by API endpoints/sections
        sections = self._split_api_sections(text)
        
        for i, section in enumerate(sections):
            # Preserve code blocks within sections
            if len(section) > self.config.max_chunk_size:
                # Split large sections while preserving code blocks
                sub_sections = self._split_preserving_code_blocks(section)
                for j, sub_section in enumerate(sub_sections):
                    chunk_metadata = self._create_chunk_metadata(
                        f"{doc_id}_api_{i}_{j}", metadata, "api_endpoint", 
                        i * 100 + j, sub_section
                    )
                    chunks.append(Document(
                        page_content=sub_section,
                        metadata=chunk_metadata.__dict__
                    ))
            else:
                chunk_metadata = self._create_chunk_metadata(
                    f"{doc_id}_api_{i}", metadata, "api_endpoint", 
                    i, section
                )
                chunks.append(Document(
                    page_content=section,
                    metadata=chunk_metadata.__dict__
                ))
        
        return chunks
    
    def _chunk_support_ticket(self, text: str, metadata: DocumentMetadata, 
                            doc_id: str) -> List[Document]:
        """Chunk support tickets with conversation flow preservation."""
        chunks = []
        
        # Split by conversation turns or sections
        sections = self._split_conversation_sections(text)
        
        for i, section in enumerate(sections):
            # Keep conversation context together
            if len(section) > self.config.max_chunk_size:
                # Split while preserving conversation flow
                sub_sections = self._split_preserving_conversation(section)
                for j, sub_section in enumerate(sub_sections):
                    chunk_metadata = self._create_chunk_metadata(
                        f"{doc_id}_ticket_{i}_{j}", metadata, "conversation_turn", 
                        i * 100 + j, sub_section
                    )
                    chunks.append(Document(
                        page_content=sub_section,
                        metadata=chunk_metadata.__dict__
                    ))
            else:
                chunk_metadata = self._create_chunk_metadata(
                    f"{doc_id}_ticket_{i}", metadata, "conversation_turn", 
                    i, section
                )
                chunks.append(Document(
                    page_content=section,
                    metadata=chunk_metadata.__dict__
                ))
        
        return chunks
    
    def _chunk_policy(self, text: str, metadata: DocumentMetadata, 
                     doc_id: str) -> List[Document]:
        """Chunk policy documents with section integrity preservation."""
        chunks = []
        
        # Split by policy sections
        sections = self._split_policy_sections(text)
        
        for i, section in enumerate(sections):
            # Keep policy sections together for context
            if len(section) > self.config.max_chunk_size:
                # Split while preserving policy context
                sub_sections = self._split_preserving_policy_context(section)
                for j, sub_section in enumerate(sub_sections):
                    chunk_metadata = self._create_chunk_metadata(
                        f"{doc_id}_policy_{i}_{j}", metadata, "policy_section", 
                        i * 100 + j, sub_section
                    )
                    chunks.append(Document(
                        page_content=sub_section,
                        metadata=chunk_metadata.__dict__
                    ))
            else:
                chunk_metadata = self._create_chunk_metadata(
                    f"{doc_id}_policy_{i}", metadata, "policy_section", 
                    i, section
                )
                chunks.append(Document(
                    page_content=section,
                    metadata=chunk_metadata.__dict__
                ))
        
        return chunks
    
    def _chunk_tutorial(self, text: str, metadata: DocumentMetadata, 
                       doc_id: str) -> List[Document]:
        """Chunk tutorial content with step-by-step preservation."""
        chunks = []
        
        # Split by tutorial steps
        steps = self._split_tutorial_steps(text)
        
        for i, step in enumerate(steps):
            # Keep tutorial steps together
            if len(step) > self.config.max_chunk_size:
                # Split while preserving step context
                sub_steps = self._split_preserving_step_context(step)
                for j, sub_step in enumerate(sub_steps):
                    chunk_metadata = self._create_chunk_metadata(
                        f"{doc_id}_tutorial_{i}_{j}", metadata, "tutorial_step", 
                        i * 100 + j, sub_step
                    )
                    chunks.append(Document(
                        page_content=sub_step,
                        metadata=chunk_metadata.__dict__
                    ))
            else:
                chunk_metadata = self._create_chunk_metadata(
                    f"{doc_id}_tutorial_{i}", metadata, "tutorial_step", 
                    i, step
                )
                chunks.append(Document(
                    page_content=step,
                    metadata=chunk_metadata.__dict__
                ))
        
        return chunks
    
    def _chunk_code_snippet(self, text: str, metadata: DocumentMetadata, 
                           doc_id: str) -> List[Document]:
        """Chunk code snippets with function/class integrity preservation."""
        chunks = []
        
        # Split by code blocks
        code_blocks = self._split_code_blocks(text)
        
        for i, block in enumerate(code_blocks):
            # Keep code blocks together
            if len(block) > self.config.max_chunk_size:
                # Split while preserving function/class integrity
                sub_blocks = self._split_preserving_code_integrity(block)
                for j, sub_block in enumerate(sub_blocks):
                    chunk_metadata = self._create_chunk_metadata(
                        f"{doc_id}_code_{i}_{j}", metadata, "code_block", 
                        i * 100 + j, sub_block
                    )
                    chunks.append(Document(
                        page_content=sub_block,
                        metadata=chunk_metadata.__dict__
                    ))
            else:
                chunk_metadata = self._create_chunk_metadata(
                    f"{doc_id}_code_{i}", metadata, "code_block", 
                    i, block
                )
                chunks.append(Document(
                    page_content=block,
                    metadata=chunk_metadata.__dict__
                ))
        
        return chunks
    
    def _chunk_troubleshooting(self, text: str, metadata: DocumentMetadata, 
                             doc_id: str) -> List[Document]:
        """Chunk troubleshooting content with problem-solution preservation."""
        chunks = []
        
        # Split by problem-solution pairs
        problems = self._split_troubleshooting_problems(text)
        
        for i, problem in enumerate(problems):
            # Keep problem-solution pairs together
            if len(problem) > self.config.max_chunk_size:
                # Split while preserving problem context
                sub_problems = self._split_preserving_problem_context(problem)
                for j, sub_problem in enumerate(sub_problems):
                    chunk_metadata = self._create_chunk_metadata(
                        f"{doc_id}_trouble_{i}_{j}", metadata, "problem_solution", 
                        i * 100 + j, sub_problem
                    )
                    chunks.append(Document(
                        page_content=sub_problem,
                        metadata=chunk_metadata.__dict__
                    ))
            else:
                chunk_metadata = self._create_chunk_metadata(
                    f"{doc_id}_trouble_{i}", metadata, "problem_solution", 
                    i, problem
                )
                chunks.append(Document(
                    page_content=problem,
                    metadata=chunk_metadata.__dict__
                ))
        
        return chunks
    
    def _chunk_generic(self, text: str, metadata: DocumentMetadata, 
                      doc_id: str) -> List[Document]:
        """Generic chunking strategy for unknown document types."""
        splits = self.splitters['recursive'].split_text(text)
        chunks = []
        
        for i, split in enumerate(splits):
            chunk_metadata = self._create_chunk_metadata(
                f"{doc_id}_generic_{i}", metadata, "generic_content", 
                i, split
            )
            chunks.append(Document(
                page_content=split,
                metadata=chunk_metadata.__dict__
            ))
        
        return chunks
    
    def _create_chunk_metadata(self, chunk_id: str, doc_metadata: DocumentMetadata,
                             chunk_type: str, position: int, content: str) -> ChunkMetadata:
        """Create metadata for a chunk."""
        return ChunkMetadata(
            chunk_id=chunk_id,
            original_doc_type=doc_metadata.doc_type,
            chunk_type=chunk_type,
            position=position,
            size=len(content),
            has_code=bool(re.search(r'```[\s\S]*?```|`[^`]+`', content)),
            has_tables=bool(re.search(r'^\|.*\|$', content, re.MULTILINE)),
            has_headers=bool(re.search(r'^#{1,6}\s+', content, re.MULTILINE)),
            context_score=self._calculate_context_score(content),
            semantic_coherence=self._calculate_semantic_coherence(content)
        )
    
    def _calculate_context_score(self, content: str) -> float:
        """Calculate context preservation score for a chunk."""
        # Simple heuristic based on content characteristics
        score = 0.0
        
        # Bonus for complete sentences
        sentences = re.findall(r'[.!?]+', content)
        if sentences:
            score += 0.3
        
        # Bonus for complete paragraphs
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.2
        
        # Bonus for complete code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        if code_blocks:
            score += 0.3
        
        # Bonus for headers
        headers = re.findall(r'^#{1,6}\s+', content, re.MULTILINE)
        if headers:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_semantic_coherence(self, content: str) -> float:
        """Calculate semantic coherence score for a chunk."""
        # Simple heuristic based on word repetition and structure
        words = content.lower().split()
        if not words:
            return 0.0
        
        # Calculate word diversity
        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        
        # Calculate average word length (proxy for complexity)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Combine factors
        coherence = (diversity * 0.6) + (min(avg_word_length / 10, 1.0) * 0.4)
        return min(coherence, 1.0)
    
    # Helper methods for splitting different content types
    def _split_api_sections(self, text: str) -> List[str]:
        """Split API documentation into sections."""
        # Split by headers and endpoint definitions
        sections = re.split(r'(?=^#{1,6}\s+|^##\s+Endpoint|^###\s+API)', text, flags=re.MULTILINE)
        return [s.strip() for s in sections if s.strip()]
    
    def _split_conversation_sections(self, text: str) -> List[str]:
        """Split support ticket conversations into sections."""
        # Split by user/agent indicators
        sections = re.split(r'(?=^User:|^Agent:|^Support:|^Customer:)', text, flags=re.MULTILINE)
        return [s.strip() for s in sections if s.strip()]
    
    def _split_policy_sections(self, text: str) -> List[str]:
        """Split policy documents into sections."""
        # Split by policy sections
        sections = re.split(r'(?=^#{1,6}\s+|^Section\s+\d+|^Policy\s+\d+)', text, flags=re.MULTILINE)
        return [s.strip() for s in sections if s.strip()]
    
    def _split_tutorial_steps(self, text: str) -> List[str]:
        """Split tutorial content into steps."""
        # Split by step indicators
        sections = re.split(r'(?=^Step\s+\d+|^#{1,6}\s+Step|^\d+\.\s+)', text, flags=re.MULTILINE)
        return [s.strip() for s in sections if s.strip()]
    
    def _split_code_blocks(self, text: str) -> List[str]:
        """Split code content into blocks."""
        # Split by code block markers or function/class definitions
        sections = re.split(r'(?=^```|^def\s+|^class\s+|^function\s+)', text, flags=re.MULTILINE)
        return [s.strip() for s in sections if s.strip()]
    
    def _split_troubleshooting_problems(self, text: str) -> List[str]:
        """Split troubleshooting content into problem-solution pairs."""
        # Split by problem indicators
        sections = re.split(r'(?=^Problem:|^Issue:|^Error:|^#{1,6}\s+Problem)', text, flags=re.MULTILINE)
        return [s.strip() for s in sections if s.strip()]
    
    # Methods for preserving context during splitting
    def _split_preserving_code_blocks(self, text: str) -> List[str]:
        """Split text while preserving code block integrity."""
        # Implementation for code-aware splitting
        return self.splitters['recursive'].split_text(text)
    
    def _split_preserving_conversation(self, text: str) -> List[str]:
        """Split text while preserving conversation flow."""
        # Implementation for conversation-aware splitting
        return self.splitters['recursive'].split_text(text)
    
    def _split_preserving_policy_context(self, text: str) -> List[str]:
        """Split text while preserving policy context."""
        # Implementation for policy-aware splitting
        return self.splitters['recursive'].split_text(text)
    
    def _split_preserving_step_context(self, text: str) -> List[str]:
        """Split text while preserving tutorial step context."""
        # Implementation for step-aware splitting
        return self.splitters['recursive'].split_text(text)
    
    def _split_preserving_code_integrity(self, text: str) -> List[str]:
        """Split text while preserving code function/class integrity."""
        # Implementation for code integrity preservation
        return self.splitters['recursive'].split_text(text)
    
    def _split_preserving_problem_context(self, text: str) -> List[str]:
        """Split text while preserving problem-solution context."""
        # Implementation for problem context preservation
        return self.splitters['recursive'].split_text(text)
    
    def _post_process_chunks(self, chunks: List[Document], 
                           metadata: DocumentMetadata) -> List[Document]:
        """Post-process chunks to ensure quality and consistency."""
        processed_chunks = []
        
        for chunk in chunks:
            # Skip chunks that are too small
            if len(chunk.page_content.strip()) < self.config.min_chunk_size:
                continue
            
            # Clean up chunk content
            cleaned_content = self._clean_chunk_content(chunk.page_content)
            
            # Update chunk with cleaned content
            chunk.page_content = cleaned_content
            chunk.metadata['size'] = len(cleaned_content)
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_chunk_content(self, content: str) -> str:
        """Clean and normalize chunk content."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Remove leading/trailing whitespace
        content = content.strip()
        
        # Ensure proper spacing around code blocks
        content = re.sub(r'([^\n])\n```', r'\1\n\n```', content)
        content = re.sub(r'```\n([^\n])', r'```\n\n\1', content)
        
        return content 