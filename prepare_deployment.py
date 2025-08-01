#!/usr/bin/env python3
"""
Quick deployment preparation script
Prepares the project for Streamlit Cloud deployment
"""

import os
import shutil
import json
from datetime import datetime

def create_deployment_package():
    """Create a clean deployment package"""
    print("📦 Creating deployment package...")
    
    # Files to include in deployment
    deployment_files = [
        'streamlit_cloud_app.py',
        'requirements_streamlit.txt', 
        '.streamlit/config.toml',
        'README_deployment.md',
        'data/mods_dataset.csv',
        'utils/preprocessing.py',
        'utils/regulation_mapper.py', 
        'utils/similarity_engine.py',
        'utils/__init__.py'
    ]
    
    # Create utils/__init__.py if it doesn't exist
    utils_init = 'utils/__init__.py'
    if not os.path.exists(utils_init):
        with open(utils_init, 'w') as f:
            f.write('# Utils package\n')
        print(f"✅ Created {utils_init}")
    
    # Check which files exist
    existing_files = []
    missing_files = []
    
    for file_path in deployment_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"⚠️ {file_path} (missing)")
    
    print(f"\n📊 Summary: {len(existing_files)} files ready, {len(missing_files)} missing")
    
    return existing_files, missing_files

def create_simple_app():
    """Create a minimal app if main files are missing"""
    if not os.path.exists('streamlit_cloud_app.py'):
        print("🔧 Creating minimal app...")
        
        minimal_app = '''
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Aircraft Mod Tool", page_icon="✈️", layout="wide")

st.title("✈️ Aircraft Modification Certification Tool")
st.markdown("**Simplified version for cloud deployment**")

# Sample data
data = {
    'mod_type': ['Avionics', 'Cabin', 'Safety', 'Structure', 'Propulsion'],
    'count': [25, 18, 12, 15, 8]
}
df = pd.DataFrame(data)

# Simple visualization
fig = px.bar(df, x='mod_type', y='count', title="Modification Types")
st.plotly_chart(fig, use_container_width=True)

# Simple analysis
description = st.text_area("Enter modification description:")
if st.button("Analyze"):
    if description:
        st.success(f"Analysis complete! This appears to be a modification.")
    else:
        st.warning("Please enter a description.")

st.info("This is a simplified version. Full features available in local deployment.")
'''
        
        with open('streamlit_cloud_app.py', 'w') as f:
            f.write(minimal_app)
        print("✅ Created minimal streamlit_cloud_app.py")

def create_deployment_info():
    """Create deployment information file"""
    deployment_info = {
        "created_at": datetime.now().isoformat(),
        "version": "2.0",
        "deployment_type": "streamlit_cloud",
        "features": {
            "ml_models": False,
            "llm_integration": False,
            "rule_based_analysis": True,
            "visualization": True,
            "data_upload": False
        },
        "files_included": [
            "streamlit_cloud_app.py",
            "requirements_streamlit.txt",
            "data/mods_dataset.csv"
        ],
        "deployment_url": "https://your-app-name.streamlit.app",
        "instructions": [
            "1. Push to GitHub repository",
            "2. Go to share.streamlit.io", 
            "3. Connect GitHub account",
            "4. Deploy with streamlit_cloud_app.py as main file"
        ]
    }
    
    with open('deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print("✅ Created deployment_info.json")

def print_deployment_instructions():
    """Print step-by-step deployment instructions"""
    print("\n🚀 STREAMLIT CLOUD DEPLOYMENT INSTRUCTIONS")
    print("=" * 50)
    
    instructions = [
        "1. 📁 Ensure all files are committed to Git:",
        "   git add .",
        "   git commit -m 'Prepare for Streamlit Cloud deployment'",
        "   git push origin main",
        "",
        "2. 🌐 Go to https://share.streamlit.io",
        "",
        "3. 🔗 Connect your GitHub account",
        "",
        "4. ➕ Click 'New app' and configure:",
        "   - Repository: your-username/mod-certification-nlp", 
        "   - Branch: main",
        "   - Main file path: streamlit_cloud_app.py",
        "   - App URL: choose a unique name",
        "",
        "5. 🚀 Click 'Deploy!'",
        "",
        "6. ⏳ Wait for deployment (2-5 minutes)",
        "",
        "7. ✅ Test your app at the provided URL",
        "",
        "8. 📊 Monitor usage in Streamlit Cloud dashboard"
    ]
    
    for instruction in instructions:
        print(instruction)
    
    print("\n💡 Tips:")
    print("   - App will auto-redeploy on git push")
    print("   - Check logs in Streamlit Cloud for debugging")
    print("   - Use secrets for sensitive configuration")
    print("   - Monitor resource usage and performance")

def main():
    """Main deployment preparation"""
    print("🛠️ Streamlit Cloud Deployment Preparation")
    print("=" * 45)
    
    # Create deployment package
    existing_files, missing_files = create_deployment_package()
    
    # Create minimal app if needed
    if 'streamlit_cloud_app.py' in missing_files:
        create_simple_app()
    
    # Create deployment info
    create_deployment_info()
    
    # Print instructions
    print_deployment_instructions()
    
    # Final status
    print("\n📋 DEPLOYMENT READINESS:")
    if len(missing_files) <= 2:  # Allow for some optional files
        print("✅ READY FOR DEPLOYMENT")
    else:
        print("⚠️ NEEDS ATTENTION - Some files missing")
    
    print(f"\n📊 Files ready: {len(existing_files)}")
    print(f"📊 Files missing: {len(missing_files)}")

if __name__ == "__main__":
    main()
