#!/usr/bin/env python3
"""
Quick start script for Indian Legal Document Search System
"""

import os
import sys
import subprocess
import platform

def check_venv():
    """Check if virtual environment is activated"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def activate_venv():
    """Activate virtual environment"""
    if platform.system() == "Windows":
        activate_script = os.path.join("venv", "Scripts", "activate")
    else:
        activate_script = os.path.join("venv", "bin", "activate")
    
    if os.path.exists(activate_script):
        print("🔧 Activating virtual environment...")
        if platform.system() == "Windows":
            os.system(f"call {activate_script}")
        else:
            os.system(f"source {activate_script}")
        return True
    return False

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import torch
        import transformers
        import sentence_transformers
        import pdfplumber
        import docx
        import spacy
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def check_models():
    """Check if required models are available"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model is available")
        return True
    except:
        print("⚠️  spaCy model not found")
        return False

def run_app():
    """Run the Streamlit application"""
    print("🚀 Starting Indian Legal Document Search System...")
    print("📱 Opening web interface...")
    
    # Run streamlit
    try:
        subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main function"""
    print("⚖️ Indian Legal Document Search System")
    print("=" * 50)
    
    # Check if virtual environment is activated
    if not check_venv():
        print("⚠️  Virtual environment not detected")
        print("🔧 Attempting to activate virtual environment...")
        if not activate_venv():
            print("❌ Failed to activate virtual environment")
            print("Please activate it manually:")
            if platform.system() == "Windows":
                print("   venv\\Scripts\\activate")
            else:
                print("   source venv/bin/activate")
            return
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Missing dependencies. Please run setup.py first")
        return
    
    # Check models
    if not check_models():
        print("⚠️  Some models may need to be downloaded on first run")
    
    # Run application
    run_app()

if __name__ == "__main__":
    main() 