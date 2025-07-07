#!/usr/bin/env python3
"""
Setup script for Indian Legal Document Search System
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please use Python 3.11 or higher")
        return False

def create_virtual_environment():
    """Create virtual environment"""
    if os.path.exists("venv"):
        print("✅ Virtual environment already exists")
        return True
    
    return run_command("python3.11 -m venv venv", "Creating virtual environment")

def activate_virtual_environment():
    """Activate virtual environment"""
    if platform.system() == "Windows":
        activate_script = os.path.join("venv", "Scripts", "activate")
    else:
        activate_script = os.path.join("venv", "bin", "activate")
    
    if os.path.exists(activate_script):
        print("✅ Virtual environment activation script found")
        return True
    else:
        print("❌ Virtual environment activation script not found")
        return False

def install_dependencies():
    """Install Python dependencies"""
    if platform.system() == "Windows":
        pip_cmd = os.path.join("venv", "Scripts", "pip")
    else:
        pip_cmd = os.path.join("venv", "bin", "pip")
    
    return run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies")

def download_spacy_model():
    """Download spaCy model"""
    if platform.system() == "Windows":
        python_cmd = os.path.join("venv", "Scripts", "python")
    else:
        python_cmd = os.path.join("venv", "bin", "python")
    
    return run_command(f"{python_cmd} -m spacy download en_core_web_sm", "Downloading spaCy model")

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "cache", "data/income_tax", "data/gst", "data/court_judgments", "data/property_law"]
    
    print("\n📁 Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def copy_env_file():
    """Copy environment file"""
    if not os.path.exists(".env"):
        if os.path.exists("env.example"):
            import shutil
            shutil.copy("env.example", ".env")
            print("✅ Created .env file from env.example")
            print("⚠️  Please edit .env file with your configuration")
        else:
            print("⚠️  env.example not found, please create .env file manually")
    else:
        print("✅ .env file already exists")

def check_models():
    """Check if required models are available"""
    print("\n🤖 Checking model availability...")
    
    # Check if transformers cache exists
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        print("✅ Hugging Face cache directory exists")
    else:
        print("⚠️  Hugging Face cache directory not found (will be created on first run)")
    
    # Check if spaCy model is installed
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model is available")
    except:
        print("⚠️  spaCy model not found (will be downloaded on first run)")

def main():
    """Main setup function"""
    print("🚀 Setting up Indian Legal Document Search System")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Activate virtual environment
    if not activate_virtual_environment():
        print("❌ Failed to activate virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download spaCy model
    if not download_spacy_model():
        print("⚠️  Failed to download spaCy model (will be downloaded on first run)")
    
    # Create directories
    create_directories()
    
    # Copy environment file
    copy_env_file()
    
    # Check models
    check_models()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your configuration")
    print("2. Activate virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("3. Run the application:")
    print("   streamlit run app.py")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 