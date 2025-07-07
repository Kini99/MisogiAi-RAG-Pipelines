#!/usr/bin/env python3
"""
Run script for the Intelligent Document Chunking System
Handles setup and starts the FastAPI application.
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up the environment for the application."""
    # Create necessary directories
    directories = [
        "uploads",
        "vector_store", 
        "metrics",
        "logs",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Load environment variables
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
    
    print("✅ Environment setup completed")

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import langchain
        import chromadb
        import transformers
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    """Main entry point."""
    print("🚀 Starting Intelligent Document Chunking System...")
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Import and run the application
    try:
        from app import app
        import uvicorn
        
        print("🌐 Starting web server...")
        print("📱 Web interface: http://localhost:8000")
        print("📚 API documentation: http://localhost:8000/docs")
        print("🔧 Press Ctrl+C to stop the server")
        
        uvicorn.run(
            app,
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", 8000)),
            reload=os.getenv("DEBUG", "false").lower() == "true"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 