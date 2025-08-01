"""
Setup script for Aircraft Modification Certification Support Tool
"""

import os
import sys
import subprocess
import importlib

def check_and_install_package(package_name, import_name=None):
    """Check if package is installed, install if not"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"⏳ Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False

def setup_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        
        nltk_downloads = [
            'punkt', 'stopwords', 'wordnet', 
            'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'
        ]
        
        print("⏳ Downloading NLTK data...")
        for dataset in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{dataset}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{dataset}')
                except LookupError:
                    try:
                        nltk.data.find(f'taggers/{dataset}')
                    except LookupError:
                        try:
                            nltk.data.find(f'chunkers/{dataset}')
                        except LookupError:
                            nltk.download(dataset, quiet=True)
                            print(f"  ✅ Downloaded {dataset}")
        
        print("✅ NLTK data setup complete")
        return True
    except Exception as e:
        print(f"❌ NLTK setup failed: {e}")
        return False

def setup_spacy_model():
    """Download spaCy English model"""
    try:
        import spacy
        try:
            spacy.load('en_core_web_sm')
            print("✅ spaCy English model already available")
        except OSError:
            print("⏳ Downloading spaCy English model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("✅ spaCy English model downloaded")
        return True
    except Exception as e:
        print(f"❌ spaCy setup failed: {e}")
        return False

def create_basic_sample_data():
    """Create basic sample data as fallback"""
    import pandas as pd
    
    # Basic sample modifications
    sample_data = [
        {
            'mod_id': 'MOD-2024-001',
            'mod_description': 'Installation of new VHF antenna on dorsal fuselage affecting structural and avionics systems for improved communication range in oceanic flights',
            'mod_type': 'Avionics',
            'regulations': 'CS 25.1309,CS 25.1431,AMC 20-22',
            'loi': 'Medium',
            'aircraft_type': 'A320',
            'approval_date': '2024-03-15'
        },
        {
            'mod_id': 'MOD-2024-002',
            'mod_description': 'Retrofit of LED cabin lighting system replacing existing fluorescent lights to improve passenger comfort and reduce power consumption',
            'mod_type': 'Cabin',
            'regulations': 'CS 25.773,CS 25.1381,AMC 25-11',
            'loi': 'Low',
            'aircraft_type': 'B737',
            'approval_date': '2024-02-20'
        },
        {
            'mod_id': 'MOD-2024-003',
            'mod_description': 'Installation of reinforced cargo door frame to increase structural integrity and support heavier cargo loads',
            'mod_type': 'Structure',
            'regulations': 'CS 25.561,CS 25.783,CS 25.807',
            'loi': 'High',
            'aircraft_type': 'A350',
            'approval_date': '2024-01-10'
        }
    ]
    
    # Create DataFrame and save
    df = pd.DataFrame(sample_data)
    df.to_csv('data/mods_dataset.csv', index=False)
    
    # Basic regulations data
    reg_data = [
        {
            'regulation_id': 'CS 25.1309',
            'title': 'Equipment systems and installations',
            'description': 'Systems and associated components must be designed to ensure safe operation',
            'category': 'Avionics',
            'applicability': 'All aircraft systems'
        },
        {
            'regulation_id': 'CS 25.773',
            'title': 'Pilot compartment view',
            'description': 'Pilot must have adequate view for safe operation',
            'category': 'Cabin',
            'applicability': 'Flight deck design'
        }
    ]
    
    reg_df = pd.DataFrame(reg_data)
    reg_df.to_csv('data/regulations_db.csv', index=False)
    
    print("✅ Basic sample data created")

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'models', 
        'notebooks',
        'app',
        'utils'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def main():
    """Main setup function"""
    print("🛫 Aircraft Modification Certification Support Tool Setup")
    print("=" * 60)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Required packages
    packages = [
        ('pandas', None),
        ('numpy', None),
        ('scikit-learn', 'sklearn'),
        ('nltk', None),
        ('spacy', None),
        ('transformers', None),
        ('sentence-transformers', 'sentence_transformers'),
        ('plotly', None),
        ('streamlit', None),
        ('matplotlib', None),
        ('seaborn', None),
        ('faiss-cpu', 'faiss'),
        ('wordcloud', None),
        ('jupyter', None)
    ]
    
    print("\n📦 Installing required packages...")
    failed_packages = []
    
    for package_name, import_name in packages:
        if not check_and_install_package(package_name, import_name):
            failed_packages.append(package_name)
    
    # Setup NLTK data
    print("\n📚 Setting up NLTK data...")
    setup_nltk_data()
    
    # Setup spaCy model
    print("\n🌐 Setting up spaCy model...")
    setup_spacy_model()
    
    # Generate sample data
    print("\n🎲 Generating sample data...")
    try:
        # Import and run sample data generation
        sys.path.append('utils')
        
        # Check if the generate_sample_data module exists
        if os.path.exists('utils/generate_sample_data.py'):
            from generate_sample_data import generate_sample_files
            generate_sample_files('data/')
            print("✅ Sample data generated successfully")
        else:
            print("⚠️  Sample data generator not found. Creating basic sample data...")
            # Create basic sample data if generator is not available
            create_basic_sample_data()
    except Exception as e:
        print(f"⚠️  Sample data generation failed: {e}")
        print("   Creating basic sample data as fallback...")
        try:
            create_basic_sample_data()
        except Exception as e2:
            print(f"❌ Fallback data creation failed: {e2}")
            print("   You can run this later using: python utils/generate_sample_data.py")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    
    if failed_packages:
        print(f"\n⚠️  Some packages failed to install: {', '.join(failed_packages)}")
        print("   Please install them manually using pip")
    
    print("\n🚀 Next steps:")
    print("1. Open and run notebooks/1_preprocessing.ipynb")
    print("2. Continue with other notebooks in sequence")
    print("3. Launch the dashboard: streamlit run app/streamlit_app.py")
    
    print("\n📚 Documentation:")
    print("- README.md contains detailed instructions")
    print("- Each notebook has comprehensive explanations")
    print("- Utils folder contains reusable components")

if __name__ == "__main__":
    main()
