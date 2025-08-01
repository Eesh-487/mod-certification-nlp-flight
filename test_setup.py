"""
Test script to validate the Aircraft Modification Certification Tool setup
"""

import sys
import os
import importlib

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    # Core packages
    packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'plotly', 'streamlit', 'sklearn', 'nltk'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            failed_imports.append(package)
    
    # Optional packages
    optional_packages = [
        ('sentence_transformers', 'sentence-transformers'),
        ('faiss', 'faiss-cpu'),
        ('spacy', 'spacy'),
        ('transformers', 'transformers')
    ]
    
    print("\n🔧 Testing optional packages...")
    for import_name, package_name in optional_packages:
        try:
            importlib.import_module(import_name)
            print(f"  ✅ {import_name}")
        except ImportError:
            print(f"  ⚠️  {import_name} (install with: pip install {package_name})")
    
    return failed_imports

def test_data_files():
    """Test if data files exist"""
    print("\n📂 Testing data files...")
    
    data_files = [
        'data/mods_dataset.csv',
        'data/regulations_db.csv'
    ]
    
    missing_files = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    return missing_files

def test_utilities():
    """Test utility modules"""
    print("\n🔧 Testing utility modules...")
    
    # Add utils to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
    
    utils = [
        'preprocessing',
        'regulation_mapper', 
        'similarity_engine',
        'generate_sample_data'
    ]
    
    failed_utils = []
    
    for util in utils:
        try:
            importlib.import_module(util)
            print(f"  ✅ {util}")
        except ImportError as e:
            print(f"  ❌ {util}: {e}")
            failed_utils.append(util)
    
    return failed_utils

def test_streamlit_app():
    """Test if Streamlit app can be imported"""
    print("\n🌐 Testing Streamlit app...")
    
    try:
        # Change to app directory
        app_path = os.path.join(os.path.dirname(__file__), 'app')
        sys.path.append(app_path)
        
        # Try to import the app module (without running it)
        import ast
        
        with open('app/streamlit_app.py', 'r') as f:
            content = f.read()
        
        # Parse the file to check for syntax errors
        ast.parse(content)
        print("  ✅ Streamlit app syntax is valid")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Streamlit app error: {e}")
        return False

def generate_sample_data_if_missing():
    """Generate sample data if missing"""
    if not os.path.exists('data/mods_dataset.csv'):
        print("\n🎲 Generating missing sample data...")
        try:
            sys.path.append('utils')
            from generate_sample_data import generate_sample_files
            generate_sample_files('data/')
            print("  ✅ Sample data generated")
            return True
        except Exception as e:
            print(f"  ❌ Failed to generate sample data: {e}")
            return False
    return True

def main():
    """Main test function"""
    print("🧪 Aircraft Modification Tool - System Test")
    print("=" * 50)
    
    # Test imports
    failed_imports = test_imports()
    
    # Generate sample data if missing
    generate_sample_data_if_missing()
    
    # Test data files
    missing_files = test_data_files()
    
    # Test utilities
    failed_utils = test_utilities()
    
    # Test Streamlit app
    app_ok = test_streamlit_app()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    
    if not failed_imports and not missing_files and not failed_utils and app_ok:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n🚀 To start the application:")
        print("   streamlit run app/streamlit_app.py")
    else:
        print("⚠️  Some issues were found:")
        
        if failed_imports:
            print(f"   📦 Missing packages: {', '.join(failed_imports)}")
            print(f"   💡 Install with: pip install {' '.join(failed_imports)}")
        
        if missing_files:
            print(f"   📂 Missing files: {', '.join(missing_files)}")
            print(f"   💡 Run: python setup.py")
        
        if failed_utils:
            print(f"   🔧 Utility issues: {', '.join(failed_utils)}")
        
        if not app_ok:
            print(f"   🌐 Streamlit app has issues")

if __name__ == "__main__":
    main()
