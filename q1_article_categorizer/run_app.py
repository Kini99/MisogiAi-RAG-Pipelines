#!/usr/bin/env python3
"""
Startup script for the Article Classification System.
Sets up environment variables to prevent threading conflicts.
"""

import os
import sys
import subprocess

def setup_environment():
    """Set environment variables to prevent threading conflicts."""
    # Disable parallel processing in various libraries
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['BLAS_NUM_THREADS'] = '1'
    os.environ['LAPACK_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    
    # Set Numba threading layer
    os.environ['NUMBA_THREADING_LAYER'] = 'tbb'
    
    # Disable warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    print("✅ Environment configured for single-threaded operation")

def main():
    """Main function to run the Streamlit app."""
    setup_environment()
    
    # Run the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    
    print("🚀 Starting Article Classification System...")
    print("📝 Use Ctrl+C to stop the application")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 