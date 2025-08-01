#!/usr/bin/env python3
"""
Test script to verify Streamlit Cloud compatibility
Run this before deployment to check for potential issues
"""

import sys
import importlib
import subprocess
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'sklearn',
        'sentence_transformers',
        'faiss',
        'nltk',
        'requests'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_data_files():
    """Test if required data files exist"""
    print("\n📁 Testing data file availability...")
    
    required_files = [
        'data/mods_dataset.csv',
        'utils/preprocessing.py',
        'utils/regulation_mapper.py',
        'utils/similarity_engine.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"⚠️ {file_path} (will use fallback)")
            missing_files.append(file_path)
    
    return missing_files

def test_streamlit_config():
    """Test Streamlit configuration"""
    print("\n⚙️ Testing Streamlit configuration...")
    
    config_file = '.streamlit/config.toml'
    if os.path.exists(config_file):
        print(f"✅ {config_file}")
    else:
        print(f"⚠️ {config_file} not found")
    
    requirements_file = 'requirements_streamlit.txt'
    if os.path.exists(requirements_file):
        print(f"✅ {requirements_file}")
    else:
        print(f"❌ {requirements_file} missing")
        return False
    
    return True

def test_app_startup():
    """Test if the app can start without errors"""
    print("\n🚀 Testing application startup...")
    
    try:
        # Try to import the main app
        sys.path.append('.')
        import streamlit_cloud_app
        print("✅ App imports successfully")
        
        # Try to create the app instance
        app = streamlit_cloud_app.CloudModCertificationApp()
        print("✅ App initializes successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ App startup failed: {e}")
        return False

def check_file_sizes():
    """Check file sizes for cloud deployment limits"""
    print("\n📏 Checking file sizes...")
    
    max_file_size = 100 * 1024 * 1024  # 100MB limit
    large_files = []
    
    for root, dirs, files in os.walk('.'):
        # Skip .git and .venv directories
        dirs[:] = [d for d in dirs if d not in ['.git', '.venv', '__pycache__']]
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                if size > max_file_size:
                    large_files.append((file_path, size))
                elif size > 10 * 1024 * 1024:  # Warn for files > 10MB
                    print(f"⚠️ Large file: {file_path} ({size / 1024 / 1024:.1f}MB)")
            except OSError:
                continue
    
    if large_files:
        print("❌ Files too large for deployment:")
        for file_path, size in large_files:
            print(f"   {file_path}: {size / 1024 / 1024:.1f}MB")
    else:
        print("✅ All files within size limits")
    
    return len(large_files) == 0

def generate_deployment_checklist():
    """Generate a deployment checklist"""
    print("\n📋 Deployment Checklist:")
    print("=" * 50)
    
    checklist = [
        "✅ Create GitHub repository",
        "✅ Push code to repository", 
        "✅ Ensure requirements_streamlit.txt is present",
        "✅ Set main file to 'streamlit_cloud_app.py'",
        "✅ Test locally with: streamlit run streamlit_cloud_app.py",
        "✅ Go to share.streamlit.io",
        "✅ Connect GitHub account",
        "✅ Deploy app with correct settings",
        "✅ Test deployed app functionality",
        "✅ Monitor app performance"
    ]
    
    for item in checklist:
        print(f"  {item}")

def main():
    """Run all compatibility tests"""
    print("🧪 Streamlit Cloud Compatibility Test")
    print("=" * 40)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run tests
    failed_imports = test_imports()
    missing_files = test_data_files()
    config_ok = test_streamlit_config()
    app_ok = test_app_startup()
    size_ok = check_file_sizes()
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 20)
    
    if not failed_imports:
        print("✅ All packages import successfully")
    else:
        print(f"❌ {len(failed_imports)} packages failed to import")
    
    if missing_files:
        print(f"⚠️ {len(missing_files)} optional files missing (app will use fallbacks)")
    else:
        print("✅ All data files present")
    
    if config_ok:
        print("✅ Streamlit configuration ready")
    else:
        print("❌ Streamlit configuration issues")
    
    if app_ok:
        print("✅ Application startup successful")
    else:
        print("❌ Application startup failed")
    
    if size_ok:
        print("✅ File sizes compatible with cloud deployment")
    else:
        print("❌ Some files too large for deployment")
    
    # Overall assessment
    critical_issues = len(failed_imports) > 0 or not config_ok or not app_ok or not size_ok
    
    if critical_issues:
        print("\n❌ DEPLOYMENT NOT RECOMMENDED")
        print("   Fix critical issues before deploying")
    else:
        print("\n✅ READY FOR DEPLOYMENT")
        print("   App should work on Streamlit Cloud")
    
    generate_deployment_checklist()
    
    return not critical_issues

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
