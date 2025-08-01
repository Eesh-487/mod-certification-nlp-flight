"""
Launch script for Aircraft Modification Certification Support Tool
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    print("🛫 Launching Aircraft Modification Certification Support Tool")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('app/streamlit_app.py'):
        print("❌ Error: streamlit_app.py not found in app/ directory")
        print("   Please run this script from the project root directory")
        return
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit is not installed")
        print("   Please run: pip install streamlit")
        return
    
    # Launch the application
    print("🚀 Starting Streamlit application...")
    print("   The application will open in your default browser")
    print("   Press Ctrl+C to stop the application")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/streamlit_app.py",
            "--server.address", "localhost",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
