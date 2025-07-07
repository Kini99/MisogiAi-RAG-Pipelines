# Intelligent Document Chunking System

An enterprise-grade document processing and knowledge management system that automatically detects document types and applies adaptive chunking strategies for optimal retrieval accuracy.

## 🚀 Features

### Core Capabilities
- **Intelligent Document Classification**: Automatically detects content types (technical docs, API references, support tickets, policies, tutorials, code snippets, troubleshooting)
- **Adaptive Chunking Strategies**: Applies document-specific chunking methods for optimal context preservation
- **LangChain Integration**: Seamless integration with LangChain for vector store management and retrieval
- **Performance Monitoring**: Comprehensive tracking of retrieval accuracy and system performance
- **Multi-format Support**: Handles PDF, DOCX, Markdown, HTML, CSV, JSON, and plain text files

### Advanced Features
- **Semantic Chunking**: Preserves semantic coherence in document chunks
- **Code-aware Processing**: Maintains code block integrity and function/class boundaries
- **Hierarchical Structure Preservation**: Respects document headers and organizational structure
- **Real-time Performance Analytics**: Track and optimize system performance
- **Web-based Interface**: User-friendly FastAPI web application

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Document       │    │   Adaptive      │
│   Processor     │───▶│   Classifier     │───▶│   Chunker       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LangChain     │    │   Performance    │    │   Vector Store  │
│   Integration   │    │   Monitor        │    │   (Chroma)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Requirements

- Python 3.11+
- 8GB+ RAM (recommended)
- OpenAI API key (optional, for advanced QA features)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd q4_document_chunking
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

## 🚀 Quick Start

### 1. Start the Application
```bash
python app.py
```

The web interface will be available at `http://localhost:8000`

### 2. Upload and Process Documents
- Use the web interface to upload documents
- Or use the API endpoint: `POST /upload`

### 3. Search and Query
- Search documents: `POST /search`
- Ask questions: `POST /qa`
- View metrics: `GET /metrics`

## 📚 API Documentation

### Document Upload
```bash
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

### Document Search
```bash
curl -X POST "http://localhost:8000/search" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"query": "your search query", "k": 5}'
```

### Question Answering
```bash
curl -X POST "http://localhost:8000/qa" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"question": "Your question here", "llm_model": "gpt-3.5-turbo"}'
```

## 🔧 Configuration

### Environment Variables
```env
# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key

# Vector Store Configuration
VECTOR_STORE_PATH=./vector_store
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNK_SIZE=2000
MIN_CHUNK_SIZE=100

# Performance Monitoring
METRICS_DIR=./metrics
```

### Chunking Configuration
```python
from src.adaptive_chunker import ChunkConfig

config = ChunkConfig(
    chunk_size=1000,
    chunk_overlap=200,
    max_chunk_size=2000,
    min_chunk_size=100,
    preserve_headers=True,
    preserve_code_blocks=True,
    preserve_tables=True,
    preserve_lists=True
)
```

## 📊 Performance Monitoring

The system includes comprehensive performance monitoring:

### Metrics Tracked
- **Search Performance**: Response times, result counts, user feedback
- **QA Performance**: Answer quality, token usage, response times
- **Processing Performance**: Success rates, chunk counts, context scores
- **Retrieval Accuracy**: Precision, recall, F1 scores

### Performance Reports
```bash
# Generate performance report
curl -X GET "http://localhost:8000/metrics"

# Get recommendations
curl -X GET "http://localhost:8000/recommendations"
```

## 🧪 Testing

### Run Tests
```bash
pytest tests/
```

### Test Coverage
```bash
pytest --cov=src tests/
```

## 📈 Usage Examples

### 1. Process Technical Documentation
```python
from src.document_processor import DocumentProcessor

processor = DocumentProcessor()
result = processor.process_document("technical_guide.md")

print(f"Document type: {result.metadata.doc_type}")
print(f"Chunks created: {len(result.chunks)}")
print(f"Context score: {result.metadata.context_score}")
```

### 2. Search with Filters
```python
from src.langchain_integration import LangChainPipeline

pipeline = LangChainPipeline()
results = pipeline.search_documents(
    query="API authentication",
    k=5,
    filter_metadata={"source_doc_type": "api_reference"}
)
```

### 3. Custom Chunking Strategy
```python
from src.adaptive_chunker import AdaptiveChunker, ChunkConfig

config = ChunkConfig(
    chunk_size=500,
    chunk_overlap=100,
    preserve_code_blocks=True
)

chunker = AdaptiveChunker(config)
chunks = chunker.chunk_document(text, metadata, "doc_id")
```

## 🔍 Document Types Supported

| Document Type | Description | Chunking Strategy |
|---------------|-------------|-------------------|
| Technical Doc | Technical documentation | Hierarchical with header preservation |
| API Reference | API documentation | Code-aware with endpoint grouping |
| Support Ticket | Customer support conversations | Conversation flow preservation |
| Policy | Policy and procedure documents | Section integrity preservation |
| Tutorial | Step-by-step guides | Step-by-step preservation |
| Code Snippet | Code examples and snippets | Function/class integrity preservation |
| Troubleshooting | Problem-solution guides | Problem-solution pair preservation |

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "app.py"]
```

### Production Considerations
- Use a production WSGI server (Gunicorn)
- Set up proper logging and monitoring
- Configure database for metrics storage
- Implement authentication and authorization
- Set up backup and recovery procedures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the performance metrics

## 🔄 Changelog

### Version 1.0.0
- Initial release
- Intelligent document classification
- Adaptive chunking strategies
- LangChain integration
- Performance monitoring
- Web interface

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Advanced semantic chunking
- [ ] Real-time collaboration features
- [ ] Advanced analytics dashboard
- [ ] Integration with external knowledge bases
- [ ] Mobile application
- [ ] Advanced security features 