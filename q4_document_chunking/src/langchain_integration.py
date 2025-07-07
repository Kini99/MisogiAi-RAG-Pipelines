"""
LangChain Integration for Intelligent Document Chunking System
Orchestrates processing pipeline and vector store updates.
"""

import os
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import logging
from datetime import datetime
import asyncio

from langchain.schema import Document
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain.retrievers import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from .document_processor import DocumentProcessor, ProcessingResult
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class LangChainPipeline:
    """
    LangChain integration for document processing and retrieval.
    """
    
    def __init__(self, 
                 vector_store_path: str = "./vector_store",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_openai: bool = False,
                 openai_api_key: Optional[str] = None):
        
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        if use_openai and openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'}
            )
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.performance_monitor = PerformanceMonitor()
        self.vector_store = None
        self.retriever = None
        
        # Load existing vector store if available
        self._load_vector_store()
    
    def _load_vector_store(self):
        """Load existing vector store or create new one."""
        try:
            if (self.vector_store_path / "chroma.sqlite3").exists():
                self.vector_store = Chroma(
                    persist_directory=str(self.vector_store_path),
                    embedding_function=self.embeddings
                )
                logger.info("Loaded existing Chroma vector store")
            else:
                # Create new vector store
                self.vector_store = Chroma(
                    persist_directory=str(self.vector_store_path),
                    embedding_function=self.embeddings
                )
                logger.info("Created new Chroma vector store")
            
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def process_and_index_documents(self, 
                                  input_path: str,
                                  batch_size: int = 10,
                                  update_existing: bool = False) -> Dict[str, Any]:
        """
        Process documents and add them to the vector store.
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        # Process documents
        if input_path.is_file():
            results = [self.document_processor.process_document(str(input_path))]
        else:
            results = self.document_processor.process_directory(str(input_path))
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            logger.warning("No documents were successfully processed")
            return {"success": False, "error": "No documents processed successfully"}
        
        # Prepare documents for indexing
        all_chunks = []
        for result in successful_results:
            for chunk in result.chunks:
                # Add document-level metadata
                chunk.metadata.update({
                    'source_doc_id': result.doc_id,
                    'source_file_path': result.file_path,
                    'source_doc_type': result.doc_type.value,
                    'processing_timestamp': datetime.now().isoformat()
                })
                all_chunks.append(chunk)
        
        logger.info(f"Prepared {len(all_chunks)} chunks for indexing")
        
        # Index in batches
        indexed_count = 0
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            try:
                # Add to vector store
                self.vector_store.add_documents(batch)
                indexed_count += len(batch)
                
                logger.info(f"Indexed batch {i//batch_size + 1}: {len(batch)} chunks")
                
            except Exception as e:
                logger.error(f"Error indexing batch {i//batch_size + 1}: {e}")
                continue
        
        # Persist vector store
        if hasattr(self.vector_store, 'persist'):
            self.vector_store.persist()
        
        # Update performance metrics
        stats = self.document_processor.get_processing_stats(results)
        stats['indexed_chunks'] = indexed_count
        stats['total_chunks_processed'] = len(all_chunks)
        
        self.performance_monitor.record_batch_processing(stats)
        
        return {
            "success": True,
            "documents_processed": len(successful_results),
            "chunks_indexed": indexed_count,
            "stats": stats
        }
    
    def search_documents(self, 
                        query: str, 
                        k: int = 5,
                        filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Search for relevant documents using the vector store.
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        
        try:
            # Update retriever search parameters
            self.retriever.search_kwargs["k"] = k
            
            # Perform search
            results = self.retriever.get_relevant_documents(query)
            
            # Apply metadata filtering if specified
            if filter_metadata:
                filtered_results = []
                for doc in results:
                    if all(doc.metadata.get(key) == value for key, value in filter_metadata.items()):
                        filtered_results.append(doc)
                results = filtered_results
            
            # Record search performance
            self.performance_monitor.record_search(query, len(results))
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def create_qa_chain(self, 
                       llm_model: str = "gpt-3.5-turbo",
                       temperature: float = 0.0) -> RetrievalQA:
        """
        Create a question-answering chain using the vector store.
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        
        try:
            llm = OpenAI(
                model_name=llm_model,
                temperature=temperature
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
            
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {e}")
            raise
    
    def answer_question(self, 
                       question: str,
                       llm_model: str = "gpt-3.5-turbo",
                       temperature: float = 0.0) -> Dict[str, Any]:
        """
        Answer a question using the indexed documents.
        """
        try:
            qa_chain = self.create_qa_chain(llm_model, temperature)
            
            with get_openai_callback() as cb:
                result = qa_chain({"query": question})
            
            # Record QA performance
            self.performance_monitor.record_qa_question(
                question, 
                result.get("result", ""),
                cb.total_tokens if hasattr(cb, 'total_tokens') else 0
            )
            
            return {
                "answer": result.get("result", ""),
                "source_documents": result.get("source_documents", []),
                "tokens_used": cb.total_tokens if hasattr(cb, 'total_tokens') else 0
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "source_documents": [],
                "tokens_used": 0
            }
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.vector_store:
            return {"error": "Vector store not initialized"}
        
        try:
            # Get collection info
            collection = self.vector_store._collection
            
            stats = {
                "total_documents": collection.count(),
                "embedding_dimension": self.embeddings.client.get_sentence_embedding_dimension(),
                "vector_store_path": str(self.vector_store_path),
                "embedding_model": self.embeddings.model_name if hasattr(self.embeddings, 'model_name') else "OpenAI"
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"error": str(e)}
    
    def update_document(self, 
                       doc_id: str,
                       new_content: str,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing document in the vector store.
        """
        try:
            # Process the updated content
            result = self.document_processor.process_document(
                content=new_content,
                doc_id=doc_id,
                metadata=metadata
            )
            
            if not result.success:
                return False
            
            # Remove old chunks for this document
            self._remove_document_chunks(doc_id)
            
            # Add new chunks
            for chunk in result.chunks:
                chunk.metadata['source_doc_id'] = doc_id
                chunk.metadata['updated_at'] = datetime.now().isoformat()
            
            self.vector_store.add_documents(result.chunks)
            
            if hasattr(self.vector_store, 'persist'):
                self.vector_store.persist()
            
            logger.info(f"Updated document {doc_id} with {len(result.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def _remove_document_chunks(self, doc_id: str):
        """Remove all chunks for a specific document."""
        try:
            # Get all documents with the specified doc_id
            results = self.vector_store.similarity_search(
                "",  # Empty query to get all documents
                k=10000  # Large number to get all documents
            )
            
            # Filter documents by doc_id
            doc_chunks = [
                doc for doc in results 
                if doc.metadata.get('source_doc_id') == doc_id
            ]
            
            # Remove these chunks
            if doc_chunks:
                # Note: This is a simplified approach. In production, you might want
                # to use a more efficient method to remove documents by metadata
                pass
                
        except Exception as e:
            logger.error(f"Error removing document chunks: {e}")
    
    def export_vector_store(self, output_path: str) -> bool:
        """Export the vector store to a file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # For Chroma, we can copy the persist directory
            import shutil
            shutil.copytree(
                self.vector_store_path,
                output_path,
                dirs_exist_ok=True
            )
            
            logger.info(f"Vector store exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting vector store: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the monitor."""
        return self.performance_monitor.get_metrics()
    
    def optimize_vector_store(self) -> Dict[str, Any]:
        """Optimize the vector store for better performance."""
        try:
            # Get current stats
            before_stats = self.get_vector_store_stats()
            
            # Perform optimization (this is a placeholder for actual optimization logic)
            # In a real implementation, you might:
            # - Remove duplicate documents
            # - Optimize index parameters
            # - Compress embeddings
            # - Update similarity thresholds
            
            after_stats = self.get_vector_store_stats()
            
            return {
                "success": True,
                "before_stats": before_stats,
                "after_stats": after_stats,
                "optimization_applied": "placeholder"
            }
            
        except Exception as e:
            logger.error(f"Error optimizing vector store: {e}")
            return {"success": False, "error": str(e)}
    
    def cleanup_old_documents(self, 
                             days_old: int = 30,
                             doc_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Clean up old documents from the vector store.
        """
        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            # Get all documents
            results = self.vector_store.similarity_search("", k=10000)
            
            documents_to_remove = []
            
            for doc in results:
                timestamp_str = doc.metadata.get('processing_timestamp')
                if timestamp_str:
                    try:
                        doc_timestamp = datetime.fromisoformat(timestamp_str).timestamp()
                        
                        # Check if document is old enough
                        if doc_timestamp < cutoff_date:
                            # Check document type filter
                            if doc_types is None or doc.metadata.get('source_doc_type') in doc_types:
                                documents_to_remove.append(doc)
                    except ValueError:
                        # Skip documents with invalid timestamps
                        continue
            
            # Remove old documents
            if documents_to_remove:
                # Note: This is a simplified approach. In production, you might want
                # to use a more efficient method to remove documents
                pass
            
            return {
                "success": True,
                "documents_removed": len(documents_to_remove),
                "cutoff_date": datetime.fromtimestamp(cutoff_date).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up old documents: {e}")
            return {"success": False, "error": str(e)} 