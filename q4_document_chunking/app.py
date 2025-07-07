"""
FastAPI Application for Intelligent Document Chunking System
Provides web interface for document processing and retrieval.
"""

import os
import shutil
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from src.document_processor import DocumentProcessor, ProcessingResult
from src.langchain_integration import LangChainPipeline
from src.performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Document Chunking System",
    description="Enterprise knowledge management system with adaptive document chunking",
    version="1.0.0"
)

# Initialize components
document_processor = DocumentProcessor()
langchain_pipeline = LangChainPipeline()
performance_monitor = PerformanceMonitor()

# Create necessary directories
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic models for API requests/responses
class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    doc_id: Optional[str] = None
    chunks_count: Optional[int] = None
    processing_time: Optional[float] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_results: int
    search_time: float


class QARequest(BaseModel):
    question: str
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.0


class QAResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]
    tokens_used: int
    response_time: float


class ProcessingStatsResponse(BaseModel):
    total_documents: int
    successful_documents: int
    failed_documents: int
    success_rate: float
    total_chunks: int
    avg_chunks_per_doc: float
    document_types: Dict[str, int]


class PerformanceMetricsResponse(BaseModel):
    search: Dict[str, Any]
    qa: Dict[str, Any]
    processing: Dict[str, Any]
    retrieval: Dict[str, Any]
    summary: Dict[str, Any]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Intelligent Document Chunking System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .section { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, textarea, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
            button { background: #667eea; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #5a6fd8; }
            .results { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px; }
            .metric { display: inline-block; margin: 10px; padding: 10px; background: #e9ecef; border-radius: 5px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #667eea; }
            .metric-label { color: #666; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Intelligent Document Chunking System</h1>
                <p>Enterprise knowledge management with adaptive document processing</p>
            </div>
            
            <div class="section">
                <h2>Document Upload & Processing</h2>
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="file">Select Document:</label>
                        <input type="file" id="file" name="file" accept=".txt,.md,.pdf,.docx,.html,.csv,.json" required>
                    </div>
                    <button type="submit">Upload & Process</button>
                </form>
                <div id="uploadResults" class="results" style="display: none;"></div>
            </div>
            
            <div class="section">
                <h2>Document Search</h2>
                <form id="searchForm">
                    <div class="form-group">
                        <label for="searchQuery">Search Query:</label>
                        <input type="text" id="searchQuery" name="query" placeholder="Enter your search query..." required>
                    </div>
                    <div class="form-group">
                        <label for="searchK">Number of Results:</label>
                        <select id="searchK" name="k">
                            <option value="3">3</option>
                            <option value="5" selected>5</option>
                            <option value="10">10</option>
                        </select>
                    </div>
                    <button type="submit">Search</button>
                </form>
                <div id="searchResults" class="results" style="display: none;"></div>
            </div>
            
            <div class="section">
                <h2>Question Answering</h2>
                <form id="qaForm">
                    <div class="form-group">
                        <label for="qaQuestion">Question:</label>
                        <textarea id="qaQuestion" name="question" rows="3" placeholder="Ask a question about your documents..." required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="qaModel">LLM Model:</label>
                        <select id="qaModel" name="llm_model">
                            <option value="gpt-3.5-turbo" selected>GPT-3.5 Turbo</option>
                            <option value="gpt-4">GPT-4</option>
                        </select>
                    </div>
                    <button type="submit">Get Answer</button>
                </form>
                <div id="qaResults" class="results" style="display: none;"></div>
            </div>
            
            <div class="section">
                <h2>System Performance</h2>
                <div id="performanceMetrics">
                    <div class="metric">
                        <div class="metric-value" id="totalDocs">-</div>
                        <div class="metric-label">Total Documents</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="totalChunks">-</div>
                        <div class="metric-label">Total Chunks</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="successRate">-</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="avgResponseTime">-</div>
                        <div class="metric-label">Avg Response Time</div>
                    </div>
                </div>
                <button onclick="loadPerformanceMetrics()">Refresh Metrics</button>
            </div>
        </div>
        
        <script>
            // Upload form handling
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData();
                const fileInput = document.getElementById('file');
                formData.append('file', fileInput.files[0]);
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    
                    const resultsDiv = document.getElementById('uploadResults');
                    resultsDiv.style.display = 'block';
                    resultsDiv.innerHTML = `
                        <h3>Upload Results</h3>
                        <p><strong>Status:</strong> ${result.success ? 'Success' : 'Failed'}</p>
                        <p><strong>Message:</strong> ${result.message}</p>
                        ${result.chunks_count ? `<p><strong>Chunks Created:</strong> ${result.chunks_count}</p>` : ''}
                        ${result.processing_time ? `<p><strong>Processing Time:</strong> ${result.processing_time.toFixed(2)}s</p>` : ''}
                    `;
                } catch (error) {
                    console.error('Upload error:', error);
                }
            });
            
            // Search form handling
            document.getElementById('searchForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const query = document.getElementById('searchQuery').value;
                const k = document.getElementById('searchK').value;
                
                try {
                    const response = await fetch('/search', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query, k: parseInt(k)})
                    });
                    const result = await response.json();
                    
                    const resultsDiv = document.getElementById('searchResults');
                    resultsDiv.style.display = 'block';
                    resultsDiv.innerHTML = `
                        <h3>Search Results (${result.total_results} found)</h3>
                        <p><strong>Search Time:</strong> ${result.search_time.toFixed(2)}s</p>
                        ${result.results.map((doc, i) => `
                            <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                                <h4>Result ${i + 1}</h4>
                                <p><strong>Content:</strong> ${doc.page_content.substring(0, 200)}...</p>
                                <p><strong>Source:</strong> ${doc.metadata.source_file_path || 'Unknown'}</p>
                                <p><strong>Type:</strong> ${doc.metadata.source_doc_type || 'Unknown'}</p>
                            </div>
                        `).join('')}
                    `;
                } catch (error) {
                    console.error('Search error:', error);
                }
            });
            
            // QA form handling
            document.getElementById('qaForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const question = document.getElementById('qaQuestion').value;
                const model = document.getElementById('qaModel').value;
                
                try {
                    const response = await fetch('/qa', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question, llm_model: model})
                    });
                    const result = await response.json();
                    
                    const resultsDiv = document.getElementById('qaResults');
                    resultsDiv.style.display = 'block';
                    resultsDiv.innerHTML = `
                        <h3>Answer</h3>
                        <p><strong>Response Time:</strong> ${result.response_time.toFixed(2)}s</p>
                        <p><strong>Tokens Used:</strong> ${result.tokens_used}</p>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
                            <p><strong>Answer:</strong></p>
                            <p>${result.answer}</p>
                        </div>
                        <h4>Source Documents (${result.source_documents.length})</h4>
                        ${result.source_documents.map((doc, i) => `
                            <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                                <p><strong>Source ${i + 1}:</strong> ${doc.page_content.substring(0, 150)}...</p>
                            </div>
                        `).join('')}
                    `;
                } catch (error) {
                    console.error('QA error:', error);
                }
            });
            
            // Load performance metrics
            async function loadPerformanceMetrics() {
                try {
                    const response = await fetch('/metrics');
                    const metrics = await response.json();
                    
                    document.getElementById('totalDocs').textContent = metrics.summary?.total_operations || 0;
                    document.getElementById('totalChunks').textContent = metrics.processing?.total_chunks || 0;
                    document.getElementById('successRate').textContent = 
                        metrics.processing?.success_rate ? (metrics.processing.success_rate * 100).toFixed(1) + '%' : 'N/A';
                    document.getElementById('avgResponseTime').textContent = 
                        metrics.search?.avg_response_time ? metrics.search.avg_response_time.toFixed(2) + 's' : 'N/A';
                } catch (error) {
                    console.error('Error loading metrics:', error);
                }
            }
            
            // Load metrics on page load
            loadPerformanceMetrics();
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        # Validate file type
        allowed_extensions = {'.txt', '.md', '.pdf', '.docx', '.html', '.csv', '.json'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        result = document_processor.process_document(str(file_path))
        
        if result.success:
            # Add to vector store
            langchain_result = langchain_pipeline.process_and_index_documents(str(file_path))
            
            return DocumentUploadResponse(
                success=True,
                message=f"Document processed successfully",
                doc_id=result.doc_id,
                chunks_count=len(result.chunks),
                processing_time=result.processing_time
            )
        else:
            return DocumentUploadResponse(
                success=False,
                message=f"Document processing failed: {result.error_message}"
            )
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for documents."""
    try:
        start_time = datetime.now()
        
        results = langchain_pipeline.search_documents(
            query=request.query,
            k=request.k,
            filter_metadata=request.filter_metadata
        )
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Convert results to serializable format
        serializable_results = []
        for doc in results:
            serializable_results.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        
        return SearchResponse(
            results=serializable_results,
            total_results=len(results),
            search_time=search_time
        )
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qa", response_model=QAResponse)
async def answer_question(request: QARequest):
    """Answer a question using the indexed documents."""
    try:
        start_time = datetime.now()
        
        result = langchain_pipeline.answer_question(
            question=request.question,
            llm_model=request.llm_model,
            temperature=request.temperature
        )
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Convert source documents to serializable format
        serializable_sources = []
        for doc in result.get('source_documents', []):
            serializable_sources.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        
        return QAResponse(
            answer=result.get('answer', ''),
            source_documents=serializable_sources,
            tokens_used=result.get('tokens_used', 0),
            response_time=response_time
        )
    
    except Exception as e:
        logger.error(f"QA error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics():
    """Get system performance metrics."""
    try:
        metrics = performance_monitor.get_metrics()
        return PerformanceMetricsResponse(**metrics)
    
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=ProcessingStatsResponse)
async def get_processing_stats():
    """Get document processing statistics."""
    try:
        stats = langchain_pipeline.get_vector_store_stats()
        
        # Get processing metrics
        processing_metrics = performance_monitor.get_metrics(metric_type='processing')
        
        return ProcessingStatsResponse(
            total_documents=processing_metrics.get('processing', {}).get('total_documents', 0),
            successful_documents=processing_metrics.get('processing', {}).get('total_documents', 0),  # Simplified
            failed_documents=0,  # Would need to track failures separately
            success_rate=processing_metrics.get('processing', {}).get('success_rate', 0),
            total_chunks=processing_metrics.get('processing', {}).get('total_chunks', 0),
            avg_chunks_per_doc=processing_metrics.get('processing', {}).get('avg_chunk_count', 0),
            document_types=processing_metrics.get('processing', {}).get('document_types', {})
        )
    
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations")
async def get_recommendations():
    """Get performance improvement recommendations."""
    try:
        recommendations = performance_monitor.get_recommendations()
        return {"recommendations": recommendations}
    
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-directory")
async def process_directory(directory_path: str, background_tasks: BackgroundTasks):
    """Process all documents in a directory (background task)."""
    try:
        # Add to background tasks
        background_tasks.add_task(langchain_pipeline.process_and_index_documents, directory_path)
        
        return {
            "success": True,
            "message": f"Directory processing started for: {directory_path}",
            "task_id": f"dir_process_{datetime.now().timestamp()}"
        }
    
    except Exception as e:
        logger.error(f"Directory processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 